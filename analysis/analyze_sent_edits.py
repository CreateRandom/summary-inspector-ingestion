import pandas as pd
from pymongo import MongoClient
from sshtunnel import SSHTunnelForwarder
from tqdm import tqdm

from analysis.generate_cards import render_summary_to_card, draw_sample
import matplotlib.pyplot as plt

from creds.creds import ATLAS_LOGIN, ATLAS_PW, ATLAS_ADDRESS


def analyze_edits(collection):
    entries = collection.find({})
    states = []
    sent_edits = []
    for entry in entries:
        analysis = entry['analysis']
        for sent in analysis:
            sent_edits.append(sent)
        states.append(entry['summary_state'])

    edits = pd.DataFrame(sent_edits)
    edits.type.value_counts().plot(kind='bar', title=f'Sentence-wise types for {collection_name}')

    plt.show()

    states = pd.DataFrame(states)
    states[0].value_counts().plot(kind='bar', title=f'Summary-wise types for {collection_name}')
    deleted = []
    deletions = edits['deletions']

    for deletion in deletions:
        if isinstance(deletion,list):
            for elem in deletion:
                deleted.append(' '.join(elem['deleted']))
    plt.show()


if __name__ == '__main__':
    selected_samples = {}
    all_selected = []
    local = True

    if local:
        db_cred_dict = {'host': 'localhost', 'port': 27017}
        client = MongoClient(**db_cred_dict)
    else:
        mongo_login_string = 'mongodb+srv://' + ATLAS_LOGIN + ':' + ATLAS_PW + '@' + ATLAS_ADDRESS + 'test?retryWrites=true&w=majority'
        print(f'Logging into db using {mongo_login_string}')
        client = MongoClient(mongo_login_string)
    db = client['inspector']

    for collection_name in ['chen', 'see', 'presumm', 'lm']:
        collection = db[collection_name]
        analyze_edits(collection)
