import argparse
import os
import pickle

from pymongo import MongoClient

from creds.creds import ATLAS_LOGIN, ATLAS_PW, ATLAS_ADDRESS
from ingestion.ingest_canned_summaries import analyze_summary_sents, simple_sent_tokenize, classify_summary


def load_mapping(mapping_path):
    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
    return mapping

def postprocess_summary_line(line):
    # apostrophe s should not be lead by space
    line = line.replace(' \'s', '\'s')
    line = line.replace('-lrb-', '(')
    line = line.replace('-rrb-', ')')
    return line

def insert_summary_into_collection(db, collection_name, story_id, local_id, summary_lines):
    # we get some lines directly from the file
    # do some postprocesing
    summary_lines = [postprocess_summary_line(line) for line in summary_lines]

    # get the article text
    arts = db['arts']

    story_id = story_id.rstrip('.story')
    article = arts.find_one({'_id': story_id})

    if article is None:
        raise RuntimeError('Could not find article for id {}'.format(story_id))
    # TODO decide whether any additional processing is needed here
    # e.g. lowercasing the text
    article_text = article['article'].lower()

    art_sents = simple_sent_tokenize(article_text)

    analysis = analyze_summary_sents(art_sents, summary_lines)
    summary_state = classify_summary(analysis)

    # merge the summary_lines into a single text for compat with the inspector
    summary_text = '\n\n'.join(summary_lines)
    # name it after the collection for legacy purposes

    local_id_name = collection_name + '_id'
    summary_dict = {'_id': story_id, 'summary_lines': summary_lines, local_id_name: local_id, 'analysis': analysis,
                    'summary_state': summary_state, collection_name: summary_text}
    db[collection_name].insert_one(summary_dict)


def ingest_presumm(db, presumm_path, presumm_mapping_path):
    mapping = load_mapping(presumm_mapping_path)

    with open(presumm_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # sometimes, two spaces are generated instead of the separator token, manual fix for now
        line = line.replace('  ', '<q>')
        sents = line.split('<q>')
        sents = [sent.rstrip() for sent in sents]
        story_id = mapping[i]
        insert_summary_into_collection(db, 'presumm', story_id,
                                       local_id='presumm_' + str(i), summary_lines=sents)


def ingest_from_path(db, method_name, path, mapping_path, extension, file_to_id_func):
    mapping = load_mapping(mapping_path)

    for file in os.listdir(path):
        if file.endswith(extension):
            file_id = file_to_id_func(file)
            story_id = mapping[file_id]
            with open(os.path.join(path,file), 'r') as f:
                lines = f.readlines()
                lines = [line.rstrip() for line in lines]
                insert_summary_into_collection(db, method_name, story_id,
                                               local_id = method_name + '_' + str(file_id), summary_lines = lines)

def ingest_chen(db, chen_path, chen_mapping_path):
    file_to_id_func = lambda x: int(x.split('.')[0])
    ingest_from_path(db, 'chen', chen_path, chen_mapping_path, '.dec', file_to_id_func)

def ingest_see(db, see_path, see_mapping_path):
    file_to_id_func = lambda x: int(x.split('_')[0])
    ingest_from_path(db, 'see', see_path, see_mapping_path, '.txt', file_to_id_func)

def ingest_lm(db, lm_path, lm_mapping_path):
    file_to_id_func = lambda x: int(x.split('.')[1])
    ingest_from_path(db, 'lm', lm_path, lm_mapping_path, '.txt', file_to_id_func)

if __name__ == '__main__':

    local = False

    if local:
        db_cred_dict = {'host': 'localhost', 'port': 27017}
        client = MongoClient(**db_cred_dict)
    else:
        mongo_login_string = 'mongodb+srv://' + ATLAS_LOGIN + ':' + ATLAS_PW + '@' + ATLAS_ADDRESS + 'test?retryWrites=true&w=majority'
        client = MongoClient(mongo_login_string)
    db = client["final_comp"]

    ingest_presumm(db, '/home/klux/Thesis/Deployment/presumm/results/cnndm/148000.candidate', '/home/klux/Thesis/Deployment/presumm/results/cnndm/id_to_story_dict.pkl')
    ingest_lm(db, '/home/klux/Thesis/Deployment/lm/gen/', '/home/klux/Thesis/Deployment/lm/id_to_story_dict.pkl')

    ingest_see(db, '/home/klux/Thesis/Deployment/see/pointer-generator/logs/pretrained_model/decode_test_400maxenc_4beam_35mindec_100maxdec_ckpt-238410/decoded',
               '/home/klux/Thesis/Deployment/see/pointer-generator/logs/pretrained_model/decode_test_400maxenc_4beam_35mindec_100maxdec_ckpt-238410/id_to_story_dict.pkl')

    ingest_chen(db, '/home/klux/Thesis/Deployment/chen/results/output/', '/home/klux/Thesis/Deployment/chen/results/output/id_to_story_dict.pkl')