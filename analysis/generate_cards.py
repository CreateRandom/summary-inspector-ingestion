import random
import re

import jinja2
import os
import subprocess

from pymongo import MongoClient
from tqdm import tqdm

from utils.sentence_diff import word_tokenize, strip_special

def draw_sample(collection, article_ids, n=30, seed=42):
    entries = list(collection.find({}))
    random.seed(seed)
    random.shuffle(entries)
    ids = []
    for entry in entries:
        if entry['summary_state'] != 'VERBATIM' and entry['_id'] not in article_ids:
            ids.append(entry['_id'])
        if len(ids) == n:
            break
    return ids

# custom env to make tex and jinja2 compatible
latex_jinja_env = jinja2.Environment(
    block_start_string='\BLOCK{',
    block_end_string='}',
    variable_start_string='\VAR{',
    variable_end_string='}',
    comment_start_string='\#{',
    comment_end_string='}',
    line_statement_prefix='%%',
    line_comment_prefix='%#',
    trim_blocks=True,
    autoescape=False,
    loader=jinja2.FileSystemLoader(os.path.abspath('/'))
)
template_file = 'template.tex'
template = latex_jinja_env.get_template(os.path.realpath(template_file))


def render_to_card(dict, out_file):
    renderer_template = template.render(**dict)
    with open(out_file, "w") as f:  # saves tex_code to output file
        f.write(renderer_template)
    subprocess.call(['pdflatex', out_file, '--interaction=batchmode', '-aux-directory=aux'])


sent_highlight_colors = ['blue', 'cyan', 'green', 'lime', 'magenta', 'olive', 'orange', 'pink', 'purple', 'red', 'teal',
                         'violet', 'yellow']


def render_summary_to_card(id, summary_collection, article_collection):
    article_text = article_collection.find_one({'_id': id})['article']
    art_sents = article_text.split('\n\n')
    art_sents = [word_tokenize(sent) for sent in art_sents]

    art_sent_dict = dict(enumerate(art_sents))

    summary = summary_collection.find_one({'_id': id})
    summary_analysis = summary['analysis']
    all_sents_referenced = set()
    new_sents = []
    for i, analysis in enumerate(summary_analysis):
        highlight_color = sent_highlight_colors[i % len(sent_highlight_colors)]
        # use a light color if this is a VERBATIM copy
        if analysis['type'] == 'VERBATIM':
            highlight_color = 'light' + highlight_color

        sent_words = analysis['sent_words']
        # for each word: which sentence it originated from
        state_vector = analysis['state_vector']
        for i, article_sent_id in enumerate(state_vector):
            if isinstance(article_sent_id, int):
                all_sents_referenced.add(article_sent_id)
                # color the word in the summary
                word_to_color = sent_words[i]
                sent_words[i] = '\\textcolor{' + highlight_color + '}{' + sent_words[i] + '}'
                # get the right sentence to color
                art_sent_to_edit = art_sent_dict[article_sent_id]
                # strip the special characters
                stripped_sent_words = [strip_special(x) for x in art_sent_to_edit]
                if word_to_color in stripped_sent_words:
                    word_id = stripped_sent_words.index(word_to_color)
                    # make sure this word has not been hightlighted before
                    if not art_sent_to_edit[word_id].startswith('\\textcolor'):
                        art_sent_to_edit[word_id] = '\\textcolor{' + highlight_color + '}{' + art_sent_to_edit[
                            word_id] + '}'
                else:
                    print(f'Missing {word_to_color} for {summary_collection}, sent_id: {article_sent_id}')

        # mark up sentences that are not copied verbatim
        if not analysis['type'] == 'VERBATIM':
            sent_words.insert(0, '$\star$')

        new_sent = ' '.join(sent_words)
        new_sents.append(new_sent)

    summary_text = '\\\\ \n'.join(new_sents)
    # get the sentence furthest into document
    max_id = max(all_sents_referenced)

    art_sents = list(art_sent_dict.values())
    art_sents = [' '.join(sent_words) for sent_words in art_sents]
    cutoff = min(max_id + 2, len(art_sents))
    art_sents = art_sents[0:cutoff]
    article_text = '\n\n'.join(art_sents)

    article_text = article_text.replace('\n\n', '. ')
    # dollar sign for comments in tex, so escape it
    article_text = article_text.replace('$', '\$')
    article_text = article_text.replace('%', '\%')
    # escape underscores within words
    article_text = re.sub(r'(.*)_(.*)', '\\1\_\\2', article_text)
    # delete trailing spaces in front of some special characters
    article_text = re.sub(r' ([,!?:’)])', '\\1', article_text)
    article_text = re.sub(r'([(] )', '\\1', article_text)
    article_text = re.sub(r' (\'s)', '\\1', article_text)

    if not os.path.exists('generated'):
        os.makedirs('generated')

    # make one extra copy for every non-verbatim sentence
    states = [x['type'] for x in summary_analysis if x['type'] != 'VERBATIM']

    for i, state in enumerate(states):

        file_name = str(id) + '_' + str(i) + '.tex'
        out_file = os.path.join('generated', file_name)

        n_copies = len(states)

        render_dict = {'article_id': id, 'article_text': article_text, 'summary_text': summary_text,
                       'color_names': sent_highlight_colors}
        if n_copies > 1:
            render_dict['num_copies'] = n_copies
            render_dict['copy_index'] = i + 1

        render_to_card(render_dict, out_file)


if __name__ == '__main__':
    selected_samples = {}
    all_selected = []

    db_cred_dict = {'host': 'localhost', 'port': 27017}

    # get client and db
    client = MongoClient(**db_cred_dict)
    db = client['inspector']

    for collection_name in ['chen', 'see', 'presumm', 'lm']:
        collection = db[collection_name]
        selected_ids = draw_sample(collection, all_selected, n=35)
        selected_samples[collection_name] = selected_ids
        all_selected.extend(selected_ids)

    print(selected_samples)
    count = 0
    parts_folders = []
    for collection_name in selected_samples.keys():
        ids = selected_samples[collection_name]
        summary_collection = db[collection_name]
        article_collection = db['articles_cnndm']

        for id in tqdm(ids):
            render_summary_to_card(id, summary_collection, article_collection)