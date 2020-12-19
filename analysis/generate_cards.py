import copy
import random
import re

import jinja2
import os
import subprocess

from pymongo import MongoClient
from tqdm import tqdm

from utils.sentence_diff import naive_word_tokenize, strip_special


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
card_template = latex_jinja_env.get_template(os.path.realpath(template_file))


def render_to_card(dict, out_file):
    renderer_template = card_template.render(**dict)
    with open(out_file, "w") as f:  # saves tex_code to output file
        f.write(renderer_template)
    subprocess.call(['pdflatex', out_file, '--interaction=batchmode', '-aux-directory=aux'])

template_file = 'sent_example_template.tex'
example_template = latex_jinja_env.get_template(os.path.realpath(template_file))

def render_to_example_box(dict):
    example_box = example_template.render(**dict)
    return example_box


default_colors = ['blue', 'cyan', 'green', 'lime', 'magenta', 'olive', 'orange', 'pink', 'purple', 'red', 'teal',
                         'violet', 'yellow']

def generate_latex_markup(id, summary_collection, article_collection, subset=None, highlight_altered=True,
                          sent_highlight_colors=None):
    if not sent_highlight_colors:
        sent_highlight_colors = default_colors
    article_text = article_collection.find_one({'_id': id})['article']
    art_sents = article_text.split('\n\n')
    art_sents = [naive_word_tokenize(sent) for sent in art_sents]
    art_sent_dict = dict(enumerate(art_sents))

    summary = summary_collection.find_one({'_id': id})
    summary_analysis = summary['analysis']
    summ_sents = summary[summary_collection.name].split('\n\n')
    summ_sents = [naive_word_tokenize(sent) for sent in summ_sents]
    summ_sent_dict = dict(enumerate(summ_sents))

    all_sents_referenced = set()
    new_summ_sents = []
    if subset:
        generator = subset
    else:
        generator = range(len(summary_analysis))
    for i in generator:
        analysis = summary_analysis[i]
        analysis = copy.deepcopy(analysis)
        highlight_color = sent_highlight_colors[i % len(sent_highlight_colors)]

        summ_sent = summ_sent_dict[i]

        highlighted_ranges = analysis['highlighted_ranges']

        for j, ranges in highlighted_ranges.items():
            id = int(j)
            # get the right sentence to color
            art_sent_to_edit = art_sent_dict[id]

            all_sents_referenced.add(id)

            for word_range in ranges:
                in_ind_sent, out_ind_sent, in_ind_doc_sent, out_ind_doc_sent = word_range

                for word_id in range(in_ind_doc_sent, out_ind_doc_sent):
                    if not art_sent_to_edit[word_id].startswith('\\textcolor'):
                        art_sent_to_edit[word_id] = '\\textcolor{' + highlight_color + '}{' + art_sent_to_edit[
                            word_id] + '}'

                for word_id in range(in_ind_sent, out_ind_sent):
                    if not summ_sent[word_id].startswith('\\textcolor'):
                        summ_sent[word_id] = '\\textcolor{' + highlight_color + '}{' + \
                                             summ_sent[word_id] + '}'

        # mark up sentences that are not copied verbatim
        if not analysis['type'] == 'VERBATIM' and highlight_altered:
            summ_sent.insert(0, '$\star$')

        new_sent = ' '.join(summ_sent)
        new_summ_sents.append(new_sent)

    art_sents = list(art_sent_dict.values())
    art_sents = [' '.join(sent_words) for sent_words in art_sents]

    return art_sents, new_summ_sents, all_sents_referenced

def sanitize_latex(string_in):
    # dollar sign for comments in tex, so escape it
    string_in = string_in.replace('$', '\$')
    string_in = string_in.replace('%', '\%')
    # escape underscores within words
    string_in = re.sub(r'(.*)_(.*)', '\\1\_\\2', string_in)
    # delete trailing spaces in front of some special characters
    string_in = re.sub(r' ([,!?:’)])', '\\1', string_in)
    string_in = re.sub(r'([(] )', '\\1', string_in)
    string_in = re.sub(r' (\'s)', '\\1', string_in)
    return string_in


def render_summary_to_card(id, summary_collection, article_collection):
    art_sents, new_summ_sents, all_sents_referenced = generate_latex_markup(id, summary_collection, article_collection)

    summary_text = '\\\\ \n'.join(new_summ_sents)
    # get the sentence furthest into document
    max_id = max(all_sents_referenced)

    cutoff = min(max_id + 2, len(art_sents))
    art_sents = art_sents[0:cutoff]
    article_text = '\n\n'.join(art_sents)
    article_text = article_text.replace('\n\n', '. ')

    article_text = sanitize_latex(article_text)

    if not os.path.exists('generated'):
        os.makedirs('generated')

    summary = summary_collection.find_one({'_id': id})
    summary_analysis = summary['analysis']
    # make one extra copy for every non-verbatim sentence
    states = [x['type'] for x in summary_analysis if x['type'] != 'VERBATIM']

    for i, state in enumerate(states):

        file_name = str(id) + '_' + str(i) + '.tex'
        out_file = os.path.join('generated', file_name)

        n_copies = len(states)

        render_dict = {'article_id': id, 'article_text': article_text, 'summary_text': summary_text,
                       'color_names': default_colors}
        if n_copies > 1:
            render_dict['num_copies'] = n_copies
            render_dict['copy_index'] = i + 1

        render_to_card(render_dict, out_file)


def render_sentence_example(id, summary_collection, article_collection, sent_index, in_context=1, out_context=1):
    if not isinstance(sent_index,list):
        sent_index = [sent_index]
    art_sents, summ_sent, all_sents_referenced \
        = generate_latex_markup(id, summary_collection, article_collection,
                                sent_highlight_colors= ['blue', 'green'],subset= sent_index, highlight_altered=False)
    summ_sents = []
    for i in range(len(sent_index)):
        summ_sents.append(sanitize_latex(summ_sent[i]))

    summ_sent = '\\\\ \n'.join(summ_sents)
    min_id = min(all_sents_referenced)
    max_id = max(all_sents_referenced)

    cut_in = max(min_id - in_context,0)
    cut_out = min(max_id + out_context, len(art_sents))

    article_text = '\n\n'.join(art_sents[cut_in:cut_out])
    article_text = article_text.replace('\n\n', '. ')

    if cut_in > 0:
        article_text = '[...] ' + article_text
    if cut_out < len(art_sents):
        article_text = article_text + ' [...]'

    article_text = sanitize_latex(article_text)

    render_dict = {'summary_sent': summ_sent, 'article_text': article_text, 'system_name' : summary_collection.name}

    return render_to_example_box(render_dict)

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