import os
import editdistance
import math
from tqdm import tqdm
import re
import pandas as pd


def remove_duplicates(l):
    return list(set(l))

def replace_characters(total):
    total = total.replace('--', '–')
    total = total.replace(').', ')')
    total = total.replace(').', ')')
    total = total.replace(';.', ';')
    total = total.replace('%.', '%')
    total = total.replace('`', '\'')
    total = total.replace('‘', '\'')
    total = total.replace('’', '\'')
    total = total.replace('£', '#')
    total = total.replace('\'.', '\'')
    total = total.replace('-lrb-', '(')
    total = total.replace('-rrb-', ')')
    total = total.replace('½', '1/2')
    total = total.replace('\n', '')
    # for preSumm
    total = total.replace('<q>', '')
    total = total.replace('.', '')
    total = total.strip()

    # these are different space characters
    total = total.replace(' ', '')
    total = total.replace(' ', '')
    return total


def load_transformer_lm_results(filename, replace_chars=False):

    lm_regex = r'Src: (.*)?\nTgt: (.*)\nGen: (.*)\n\tRouge-1: (.*)\n\tRouge-2: (.*)\n\tRouge-L: (.*)'
    refs = []
    gens = []
    with open(filename) as file:
        buffer = []
        for cnt, line in enumerate(file):
            if re.match('-{50}', line):
                all_lines = ''.join(buffer)
                print(all_lines)

                match = re.match(lm_regex, all_lines)
                if match:
                    _, tgt, gen, _,_,_ = match.groups()
                    refs.append(tgt)
                    gens.append(gen)
                buffer = []
            else:
                buffer.append(line)

    return refs, gens

def load_single_file(filename, replace_chars=False):
    with open(filename) as file:
        lines = file.readlines()
    if replace_chars:
        lines = [replace_characters(line) for line in lines]
    entries = {}
    ids = []
    for i, line in enumerate(lines):
        entries[i] = line
        ids.append(i)
    return entries, ids


def load_all_files(path_a, replace_chars=False, n = None):
    all_lines = {}
    all_ids = []

    for i, x in enumerate(os.listdir(path_a)):

        match = re.search('.*?(\d+).*', x)

        if match:
            id = match.groups()[0]
            path_to = os.path.join(path_a, x)
            with open(path_to) as file:
                lines = file.readlines()
                total = ' '.join(lines)
                if replace_chars:
                    total = ''.join(lines)
                    total = replace_characters(total)

                all_lines[id] = total
                all_ids.append(id)

        if n is not None and i +1 >= n:
            break

    return all_lines, all_ids

# transformer lm data loading

path_chen = '/home/klux/Thesis/available_outputs/chen/reference'
path_see = '/home/klux/Thesis/available_outputs/see/reference'
path_presumm = '/home/klux/Thesis/available_outputs/PreSumm/test_abs.148000.gold'
path_lm = '/home/klux/Thesis/available_outputs/transformer_lm/tgt'


chen_ref, chen_ids = load_all_files(path_chen, replace_chars=True)
see_ref, see_ids = load_all_files(path_see, replace_chars=True)
presumm_ref, presumm_ids = load_single_file(path_presumm, replace_chars=True)
lm_ref, lm_ids = load_all_files(path_lm, replace_chars=True)


print(len(chen_ref))
print(len(see_ref))
print(len(presumm_ref))
print(len(lm_ref))
items = []
# compute exact matches
for i, x in tqdm(list(see_ref.items())):
    item = {'see_id': i}
    for j, y in chen_ref.items():
        if x == y:
            item['chen_id'] = j
            del chen_ref[j]
          #  chen_ids.remove(j)
            break
    for k, z in presumm_ref.items():
        if x == z:
            item['presumm_id'] = k
            del presumm_ref[k]
           # presumm_ids.remove(k)
            break

    for l, a in lm_ref.items():
        if x == a:
            item['lm_id'] = l
            del lm_ref[l]
            # lm_ids.remove(l)
            break

    items.append(item)

# remove the matched items

# def get_dict_subset(ids, dict):
#     new_dict = {}
#     for id in ids:
#         new_dict[id] = dict[id]
#     return new_dict
#
# chen_ref = get_dict_subset(chen_ids, chen_ref)
# presumm_ref = get_dict_subset(presumm_ids, presumm_ref)
# lm_ref = get_dict_subset(lm_ids, lm_ref)

print(len(chen_ref))
print(len(see_ref))
print(len(presumm_ref))
print(len(lm_ref))


item_frame = pd.DataFrame(items)
item_frame['article_id'] = item_frame['see_id']
item_frame.set_index('article_id', inplace=True)

print(item_frame.head())

chen_missing = item_frame[pd.isna(item_frame['chen_id'])]
presumm_missing = item_frame[pd.isna(item_frame['presumm_id'])]
lm_missing = item_frame[pd.isna(item_frame['lm_id'])]
print(f'Missing {len(chen_missing)} for chen')
print(f'Missing {len(presumm_missing)} for presumm')
print(f'Missing {len(lm_missing)} for lm')

def fill_up_missing(missing_elems, ref_text_dict, column_name):
    for id, item in missing_elems.iterrows():

        current_see = see_ref[id]
        max_value = math.inf
        best_candidate = None
        best_index = None
        for j, elem_to_comp in ref_text_dict.items():
            current = editdistance.eval(current_see, elem_to_comp)
            if current < max_value:
                max_value = current
                best_candidate = elem_to_comp
                best_index = j

        item_frame.at[id, column_name] = str(best_index)

print('Filling missing chen')
fill_up_missing(chen_missing, chen_ref, 'chen_id')
print('Filling missing presumm')
fill_up_missing(presumm_missing, presumm_ref, 'presumm_id')
print('Filling missing lm')
fill_up_missing(lm_missing, lm_ref, 'lm_id')


chen_summary_path = '/home/klux/Thesis/available_outputs/chen/rnn-ext_abs_rl_rerank/decoded'
see_summary_path = '/home/klux/Thesis/available_outputs/see/pointer-gen-cov'
presumm_summary_path = '/home/klux/Thesis/available_outputs/PreSumm/test_abs.148000.candidate'
lm_summary_path = '/home/klux/Thesis/available_outputs/transformer_lm/gen'

article_path = '/home/klux/Thesis/available_outputs/articles'

chen_summaries, _ = load_all_files(chen_summary_path)
see_summaries, _ = load_all_files(see_summary_path)
presumm_summaries, _ = load_single_file(presumm_summary_path)
lm_summaries, _ = load_all_files(lm_summary_path)
# load the reference summaries from See
see_ref, _ = load_all_files(path_see)

articles, ids = load_all_files(article_path)

item_frame['article'] = item_frame.index.map(lambda x: articles[x])
# add reference summaries as well
item_frame['ref'] = item_frame.see_id.map(lambda x: see_ref[x])

item_frame['see'] = item_frame.see_id.map(lambda x: see_summaries[x])
item_frame['chen'] = item_frame.chen_id.map(lambda x: chen_summaries[x])
item_frame['presumm'] = item_frame.presumm_id.map(lambda x: presumm_summaries[int(x)])
item_frame['lm'] = item_frame.lm_id.map(lambda x: lm_summaries[x])

def prepend_name(id, name):
    # ensure int
    if isinstance(id, float):
        id = int(id)
    return name + '_' + str(id)

item_frame['see_id'] = item_frame.see_id.map(lambda x: prepend_name(x, 'see'))
item_frame['chen_id'] = item_frame.chen_id.map(lambda x: prepend_name(x, 'chen'))
item_frame['presumm_id'] = item_frame.presumm_id.map(lambda x: prepend_name(x, 'presumm'))
item_frame['lm_id'] = item_frame.lm_id.map(lambda x: prepend_name(x, 'lm'))

print(item_frame.head())

item_frame.to_json('all_summaries.json')
