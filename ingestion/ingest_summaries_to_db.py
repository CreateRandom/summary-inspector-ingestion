import re
from collections import Counter
from tqdm import tqdm
import pandas as pd
from pymongo import MongoClient
from creds.creds import ATLAS_LOGIN, ATLAS_PW, ATLAS_ADDRESS

from utils.file_utils import store_dataframe_in_db
from utils.sentence_diff import analyze_sentence

def flatten_list(l): return [item for sublist in l for item in sublist]

seed_set = {'chen': [7627, 11412, 9357, 4126, 2389, 6249, 4497, 5631, 5392, 7558, 10952, 3000, 3506, 11366, 757, 430, 10955, 8553, 1790, 6439, 1084, 9797, 4688, 10944, 4557, 11458, 9505, 7819, 715, 9957, 7144, 9401, 9761, 11418, 6880],
            'see': [7421, 968, 4043, 5632, 10671, 499, 1805, 8914, 6493, 86, 9455, 881, 6069, 5144, 8470, 3723, 9929, 10546, 9314, 6103, 3027, 7868, 6254, 9152, 807, 9447, 4803, 3305, 5778, 4140, 3832, 1670, 7735, 10381, 10249],
            'presumm': [9325, 7055, 9461, 6394, 9065, 6428, 10417, 10228, 3749, 9821, 9256, 2773, 709, 6940, 8443, 6221, 6852, 4187, 1003, 9690, 4577, 2560, 9671, 11441, 5127, 2218, 10189, 10575, 9763, 10354, 6838, 3892, 10440, 7256, 5768],
            'lm': [5802, 1154, 6464, 2359, 9454, 4095, 3026, 6277, 10809, 4700, 5267, 4641, 6264, 8881, 436, 852, 10761, 3753, 5700, 3899, 461, 2029, 9004, 8945, 3500, 4475, 4996, 1157, 6649, 618, 1915, 6821, 291, 3988, 3578]}
ids_in_seed_set = flatten_list((seed_set.values()))

def simple_sent_tokenize(text):
    sents = text.split('\n\n')
    sents = [sent for sent in sents if sent]
    return sents

def analyze_sents(article_text, summary_text):
    results = []
    article_sents = simple_sent_tokenize(article_text)
    summary_sents = simple_sent_tokenize(summary_text)
    for sent in summary_sents:
        analysis = analyze_sentence(sent, article_sents)
        results.append(analysis)

    return results

def generate_analysis(row, column_to_analyze):
    article_text = row['article']
    summary_text = row[column_to_analyze]
    return analyze_sents(article_text, summary_text)

def classify_summary(column_with_sents):
    types = []
    for sent_analysis in column_with_sents:
        types.append(sent_analysis['type'])

    counts = Counter(types)
    # more than one type of operation
    if len(counts) > 1:
        return 'MIXED'
    # only a single type of operation, return that type
    else:
        return list(counts.keys())[0]


def get_all_sents(sents_analysis_series):
    list_sents = sents_analysis_series.to_list()
    return flatten_list(list_sents)

def prepare_data_for_db(path_to_json, db):
    tqdm.pandas()

    # load the summary data created with compile_model_summaries
    summary_frame = pd.read_json(path_to_json)

 #   summary_frame = summary_frame[0:3]
    summary_frame['article'] = summary_frame['article'].apply(lambda x: x.replace(' . ', '.\n\n'))
    summary_frame['article'] = summary_frame['article'].apply(lambda x: x.replace('? ', '?\n\n'))
    summary_frame['article'] = summary_frame['article'].apply(lambda x: x.replace('! ', '!\n\n'))

    # split into three separate frames (one for each summary type)
    # TODO think about all the replacements
    rep_func = lambda x: re.sub(r' (.)\n', r'\1\n\n',x)
    summary_frame['see'] = summary_frame['see'].apply(rep_func)
    summary_frame['presumm'] = summary_frame['presumm'].apply(lambda x: x.replace('<q>', '\n\n'))
    # sometimes, two spaces are generated instead of the separator token, manual fix for now
    summary_frame['presumm'] = summary_frame['presumm'].apply(lambda x: x.replace('  ', '\n\n'))
    summary_frame['chen'] = summary_frame['chen'].apply(rep_func)
    summary_frame['lm'] = summary_frame['lm'].apply(rep_func)

    left = lambda x: re.sub(r'-lrb-', '(',x)
    right = lambda x: re.sub(r'-rrb-', ')', x)
    summary_frame['lm'] = summary_frame['lm'].apply(left)
    summary_frame['lm'] = summary_frame['lm'].apply(right)

    # SEE
    summary_frame['see_sents_anaylsis'] = summary_frame.progress_apply(func=(lambda row: generate_analysis(row,'see')), axis=1)
    summary_frame['see_state'] = summary_frame['see_sents_anaylsis'].progress_apply(func=classify_summary)
    see = summary_frame[['see_id', 'see', 'see_sents_anaylsis', 'see_state']].copy()
    # small hack to have a consistent id scheme
    see['_id'] = see.index
    see.rename(columns={'see_sents_anaylsis': 'analysis', 'see_state' : 'summary_state'}, inplace=True)
    store_dataframe_in_db(see, db.see)
    del see
    # CHEN
    summary_frame['chen_sents_anaylsis'] = summary_frame.progress_apply(func=(lambda row: generate_analysis(row, 'chen')), axis=1)
    summary_frame['chen_state'] = summary_frame['chen_sents_anaylsis'].progress_apply(func=classify_summary)
    chen = summary_frame[['chen_id', 'chen', 'chen_sents_anaylsis', 'chen_state']].copy()
    chen['_id'] = chen.index
    chen.rename(columns={'chen_sents_anaylsis': 'analysis', 'chen_state' : 'summary_state'}, inplace=True)
    store_dataframe_in_db(chen, db.chen)
    del chen
    # PRESUMM
    summary_frame['presumm_sents_analysis'] = summary_frame.progress_apply(func=(lambda row: generate_analysis(row, 'presumm')), axis=1)
    summary_frame['presumm_state'] = summary_frame['presumm_sents_analysis'].progress_apply(func=classify_summary)
    presumm = summary_frame[['presumm_id', 'presumm', 'presumm_sents_analysis', 'presumm_state']].copy()
    presumm['_id'] = presumm.index
    presumm.rename(columns={'presumm_sents_analysis': 'analysis', 'presumm_state' : 'summary_state'}, inplace=True)
    store_dataframe_in_db(presumm, db.presumm)
    del presumm
    # LM
    summary_frame['lm_sents_analysis'] = summary_frame.progress_apply(func=(lambda row: generate_analysis(row, 'lm')), axis=1)
    summary_frame['lm_state'] = summary_frame['lm_sents_analysis'].progress_apply(func=classify_summary)
    lm = summary_frame[['lm_id', 'lm', 'lm_sents_analysis', 'lm_state']].copy()
    lm['_id'] = lm.index
    lm.rename(columns={'lm_sents_analysis': 'analysis', 'lm_state' : 'summary_state'}, inplace=True)
    store_dataframe_in_db(lm, db.lm)
    del lm

    # and one frame for the article content
    article_frame = summary_frame[['article']].copy()
    article_frame['_id'] = article_frame.index

    def get_set_type(_id):

        if _id in ids_in_seed_set:
            return 'seed'
        else:
            return 'unassigned'

    # mark articles that were part of the seed set
    article_frame['set'] = article_frame['_id'].apply(func=get_set_type)
    store_dataframe_in_db(article_frame, db.articles_cnndm)

if __name__ == '__main__':

    local = False
    if local:
        db_cred_dict = {'host': 'localhost', 'port': 27017}
        client = MongoClient(**db_cred_dict)
    else:
        mongo_login_string = 'mongodb+srv://' + ATLAS_LOGIN + ':' + ATLAS_PW + '@' + ATLAS_ADDRESS + 'test?retryWrites=true&w=majority'
        print(f'Logging into db using {mongo_login_string}')
        client = MongoClient(mongo_login_string)
    db = client['inspector']
    prepare_data_for_db('/home/klux/summary-inspector/ingestion/all_summaries.json',db)
