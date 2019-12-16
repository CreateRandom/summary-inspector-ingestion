from pymongo import MongoClient

from analysis.generate_cards import render_sentence_example
from creds.creds import ATLAS_LOGIN, ATLAS_PW, ATLAS_ADDRESS

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


    article_id = 6428
    system = 'presumm'
    summary_collection = db[system]
    sent_index = 1
    in_context = 1
    out_context = 1
    render_string = \
        render_sentence_example(article_id,summary_collection,article_collection,
                                sent_index, in_context = in_context,out_context = out_context)

    print(render_string)

if __name__ == '__main__':
    main()