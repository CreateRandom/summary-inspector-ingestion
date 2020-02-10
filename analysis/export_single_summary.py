from pymongo import MongoClient

from analysis.generate_cards import render_sentence_example
from creds.creds import ATLAS_LOGIN, ATLAS_PW, ATLAS_ADDRESS
import pandas as pd
def main():
    local = False
    if local:
        db_cred_dict = {'host': 'localhost', 'port': 27017}
        client = MongoClient(**db_cred_dict)
    else:
        mongo_login_string = 'mongodb+srv://' + ATLAS_LOGIN + ':' + ATLAS_PW + '@' + ATLAS_ADDRESS + 'test?retryWrites=true&w=majority'
        print(f'Logging into db using {mongo_login_string}')
        client = MongoClient(mongo_login_string)
    db = client['inspector']

    article_collection = db.articles_cnndm


    article_id = 4557
    system = 'chen'
    summary_collection = db[system]
    sent_index = 1
    in_context = 1
    out_context = 1


    df = pd.read_csv('/home/klux/summary-inspector-gui/notebooks/two_misleading_one_fine.csv')

    for i, row in df.iterrows():
        system = row['summary_id'].split('_')[0]
        summary_collection = db[system]
        sent_index = row['sent_id']
        render_string = \
            render_sentence_example(row['article_id'],summary_collection,article_collection,
                                    sent_index, in_context = in_context,out_context = out_context)

        print(render_string)

        print('')

if __name__ == '__main__':
    main()