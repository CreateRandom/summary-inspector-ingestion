import re
from collections import Counter

import editdistance
import numpy as np
import difflib

from nltk import word_tokenize, ngrams
from rouge import Rouge


def get_novel_ngrams(summary, document, n=2, language='english'):
    if not isinstance(document, list):
        document = word_tokenize(document, language)
    doc_grams = set(ngrams(document, n=n))
    if not isinstance(summary, list):
        summary = word_tokenize(summary, language)
    sent_grams = set(ngrams(summary, n=n))
    new_grams = [x[0] for x in sent_grams if x not in doc_grams]
    return new_grams

def naive_word_tokenize(string):
    words = string.split(' ')
    words = [word for word in words if word is not '']
    return words

def strip_special(text):
    return re.sub(r'[^a-zA-Z0-9 ]', '', text)


def get_sims(unaccounted_words, doc_sent_words, to_search):
    sm = difflib.SequenceMatcher()

    sims = []
    for i, sents_words in enumerate(doc_sent_words):
        if i not in to_search:
            sims.append(0)
        else:
            # compute the longest match
            sm.set_seqs(sents_words, unaccounted_words)
            a, b, length = sm.find_longest_match(alo=0, ahi=len(sents_words), blo=0, bhi=len(unaccounted_words))

            sims.append(length)

    to_return = np.array(sims)
    if sum(to_return) == 0:
        print(unaccounted_words)
    return to_return


class FlexList(list):
    # allows indexing by another list
    def __getitem__(self, keys):
        if isinstance(keys, (int, slice)): return list.__getitem__(self, keys)
        return [self[k] for k in keys]


def analyze_sentence(sentence, document_sents):
    sm = difflib.SequenceMatcher(autojunk=False)


    doc_sent_words_raw =[]
    for sent in document_sents:
        sent_words = naive_word_tokenize(sent)
        doc_sent_words_raw.append(sent_words)

    # strip all special characters
    document_sents_stripped = [strip_special(s) for s in document_sents]
    # first, tokenize the document into sentences
    document_sents_words = []
    for sent in document_sents_stripped:
        sent_words = naive_word_tokenize(sent)
        document_sents_words.append(sent_words)

    document_sents_words = FlexList(document_sents_words)


    sent_words_raw = naive_word_tokenize(sentence)
    sentence = strip_special(sentence)
    sent_words = naive_word_tokenize(sentence)
    sent_words = FlexList(sent_words)
    state_vector = ['UNASSIGNED'] * len(sent_words)

    all_words = []
    for doc_sent_words in document_sents_words:
        all_words.extend(doc_sent_words)

    # assign words that appear nowhere in the document
    novel_unigrams = get_novel_ngrams(sentence, all_words, n=1)

    if novel_unigrams:
        for i, word in enumerate(sent_words):
            if word in novel_unigrams:
                state_vector[i] = 'NEW'

    doc_sents_not_searched = list(range(len(document_sents_words)))

    # first pass, try to match the whole sentence to another sentence
    is_first_pass = True

    # ranges to highlight in the text
    highlighted_ranges = {}

    unaccounted_ranges = get_unaccounted_ranges(state_vector)

    # only break out once all words in the sentence have been accounted for
    while unaccounted_ranges:
        # pick the longest unaccounted range
        lengths = [len(x) for x in unaccounted_ranges]
        max_ind = np.argmax(lengths)

        target_range = unaccounted_ranges[max_ind]

        unaccounted_words = sent_words[target_range]
        # only search for the words that haven't be assigned
        # and only in the sentences we haven't searched in
        # find the most similar sentence
        sims = get_sims(unaccounted_words, document_sents_words, doc_sents_not_searched)

        most_similar_ind = int(np.argmax(sims))
        #  doc_sents_not_searched.remove(most_similar_ind)

        candidate_words = document_sents_words[most_similar_ind]
        # match word by word
        if is_first_pass:
            # on the first pass, match the whole sentence
            sm.set_seqs(sent_words, candidate_words)
            offset = 0
            is_first_pass = False
        else:
            sm.set_seqs(unaccounted_words, candidate_words)
            # offset of first elem in the list
            offset = target_range[0]
        matches = sm.get_matching_blocks()

        state_vector = update_state(matches, state_vector, most_similar_ind, offset)
        unaccounted_ranges = get_unaccounted_ranges(state_vector)

        ranges = unwrap_matches(matches, candidate_words)

        # now, match the word ranges to the original raw string
        # raw sent
        doc_sent_words_raw_current = doc_sent_words_raw[most_similar_ind]
        range_inds = []
        for word_range in ranges:
            # match in summary sent
            candidate_sequence_in = align_range_with_raw_sentence(word_range[0:3], sent_words_raw)
            in_ind_sent = int(candidate_sequence_in[0])
            candidate_sequence_out = align_range_with_raw_sentence(word_range[-3:], sent_words_raw)
            out_ind_sent = int(candidate_sequence_out[-1] +1)
            # match in document sent
            candidate_sequence_in = align_range_with_raw_sentence(word_range[0:3], doc_sent_words_raw_current)
            candidate_sequence_out = align_range_with_raw_sentence(word_range[-3:], doc_sent_words_raw_current)
            in_ind_doc_sent = int(candidate_sequence_in[0])
            out_ind_doc_sent = int(candidate_sequence_out[-1] +1)

        #    summ_sent_current = sent_words_raw[in_ind_sent:out_ind_sent]
       #     print(summ_sent_current)
            range_inds.append((in_ind_sent, out_ind_sent, in_ind_doc_sent, out_ind_doc_sent))

        # cast to string for saving
        new_index = str(most_similar_ind)
        if new_index not in highlighted_ranges:
            highlighted_ranges[new_index] = range_inds
        else:
            highlighted_ranges[new_index].extend(range_inds)

    result = {'state_vector': state_vector, 'sent_words': sent_words}

    # now, for each sentence, we can check whether it is
    # a) totally new
    # b) an unaltered copy of a single sentence
    # c) an edited copy of a single sentence (insertion, deletion, replacement with a NEW token)
    # d) a merger of two or more sentences
    state = classify_result(result, document_sents_words)
    state['state_vector'] = state_vector
    state['highlighted_ranges'] = highlighted_ranges

    state['rouge_with_closest'] = get_rouge_with_closest(result, document_sents_words)

    return state

def unwrap_matches(matches, sent):
    unwrapped = []
    for match in matches:
        # a: match index in the summary sentence, b: match id in the document sentence
        a, b, length = match
        if length == 0: continue

        words = sent[b:b+length]
        unwrapped.append(words)
    return unwrapped

# see here https://www.geeksforgeeks.org/combinations-from-n-arrays-picking-one-element-from-each-array/
def evaluate_index_sequences(arrays):
    # number of arrays
    n = len(arrays)
    min_score = n - 1
    best_combination = None
    current_best_score = np.inf
    max_to_check = np.inf
    checked = 0
    # to keep track of next element
    # in each of the n arrays
    indices = [0 for i in range(n)]
    while (True):
        checked = checked + 1
        stopped = False
        combination = []
        # current combination
        for i in range(n):
            value = arrays[i][indices[i]]
            # needs to be growing monotonically
            if not strictly_monotonic(combination):
                stopped = True
                continue
            else:
                combination.append(value)

        if not stopped:
            score = score_indices(combination)

            if score == min_score:
                return combination

            # update best score
            if score < current_best_score:
                current_best_score = score
                best_combination = combination

        # find the rightmost array that has more
        # elements left after the current element
        # in that array
        next = n - 1
        while (next >= 0 and
               (indices[next] + 1 >= len(arrays[next]))):
            next -= 1

        # no such array is found so no more
        # combinations left
        if (next < 0) or checked >= max_to_check:
            return best_combination

        # if found move to next element in that
        # array
        indices[next] += 1

        # for all arrays to the right of this
        # array current index again points to
        # first element
        for i in range(next + 1, n):
            indices[i] = 0

def score_indices(indices):
    if not strictly_monotonic(indices):
        return np.inf
    diff = np.diff(indices)
    sum_squared = np.sum(diff**2)
    return sum_squared

def strictly_monotonic(a):
    return np.all(np.diff(a) > 0)

def get_best_candidates(word, candidate_dict, n=5):
    scores = []
    for id, word_comp in candidate_dict.items():
        scores.append(editdistance.eval(word, word_comp))

    ind = (np.array(scores)).argsort()[:n]
   # print(f'{word}, {candidate_dict[ind[0]]}')
    return ind

def align_range_with_raw_sentence(range_words, raw_sentence):
    candidate_dict = {elem: i for elem, i in enumerate(raw_sentence)}
    best_sequence = None
    beam_width = 3

    while best_sequence is None:
        best_match_dict = {}
        for i, word in enumerate(range_words):
            id_list_2 = get_best_candidates(word, candidate_dict, n=beam_width)
            best_match_dict[i] = id_list_2

        arrays = list(best_match_dict.values())
        best_sequence = evaluate_index_sequences(arrays)
        if best_sequence is None:
         #   print(f'Could not align {range_words} to {raw_sentence} with beam {beam_width}, increasing.')
            beam_width = beam_width + 1

    return best_sequence

def update_state(matches, state_vector, sentence_id, offset=0):
    to_return = state_vector.copy()
    for match in matches:
        # a: match index in the summary sentence, b: match id in the document sentence
        a, b, length = match
        if length == 0: continue
        # update all matching words
        begin = a + offset
        end = a + length + offset
        for i in range(begin, end):
            if to_return[i] != 'UNASSIGNED': continue
            to_return[i] = sentence_id
    return to_return

def get_unaccounted_ranges(state_vector):
    ranges = []
    temp_ranges = []
    for i, state in enumerate(state_vector):
        if state == 'UNASSIGNED':
            temp_ranges.append(i)
        else:
            if temp_ranges:
                ranges.append(temp_ranges)
                temp_ranges = []
    # final flush
    if temp_ranges:
        ranges.append(temp_ranges)

    return ranges


tag_edit_mapping = {'equal': 'VERBATIM', 'delete': 'DELETION', 'insert': 'INSERTION', 'replace': 'REPLACEMENT'}


def analyze_tag_list(tag_list):
    counts = Counter(tag_list)

    if len(counts.keys()) > 1:
        return 'MIXED'
    else:
        return tag_edit_mapping[list(counts.keys())[0]]


def analyze_diff_operations(opcodes, summary_sent_words, doc_sent_words):
    deletions = []
    insertions = []
    replacements = []

    # total number of edit operations performed
    n_op = 0
    tag_list = []
    for tag, _, _, _, _ in opcodes:
        if tag is not 'equal':
            n_op = n_op + 1
            tag_list.append(tag)
    # what type of edit is this?
    if n_op == 0:
        edit_type = 'VERBATIM'
    else:
        edit_type = analyze_tag_list(tag_list)

    total_overlap = 0
    for tag, i1, i2, j1, j2 in opcodes:

        if tag == 'delete':
            #   print('Remove {} from positions {}:{}'.format(doc_sent_words[i1:i2], i1, i2))
            # TODO possibly look at the context
            deletions.append({'deleted': doc_sent_words[i1:i2]})

        elif tag == 'equal':
            length = i2 - i1
            total_overlap = total_overlap + length

        elif tag == 'insert':
            #      print('Insert {} from {}:{} of s2 into s1 at {}'.format(summary_sent_words[j1:j2], j1, j2, i1))
            inserted = summary_sent_words[j1:j2]

            # TODO also possibly get the context
            before = i1 - 1
            after = i1 + 1

            insertions.append({'inserted': inserted})

        elif tag == 'replace':
            #     print('Replace {} from {}:{} of s1 with {} from {}:{} of s2'.format(doc_sent_words[i1:i2], i1, i2,
            #                                                                         summary_sent_words[j1:j2], j1, j2))

            replaced = doc_sent_words[i1:i2]
            replacement = summary_sent_words[j1:j2]
            replacements.append({'replaced': replaced, 'replacement': replacement})

    share_of_doc_retained = float(total_overlap) / float(len(doc_sent_words))
    share_of_sent_from_doc = float(total_overlap) / float(len(summary_sent_words))

    return {'type': edit_type, 'share_of_sent_from_doc_sent': share_of_sent_from_doc,
            'share_of_doc_retained': share_of_doc_retained,
            'deletions': deletions, 'insertions': insertions, 'replacements': replacements}


def get_opcodes(summary_sent_words, doc_sent_words):
    matcher = difflib.SequenceMatcher(None, doc_sent_words, summary_sent_words)

    opcodes = matcher.get_opcodes()

    return opcodes

def get_rouge_with_closest(result, document_sents_words):
    sent_words = result['sent_words']
    state_vector = result['state_vector']

    only_ids = [x for x in state_vector if isinstance(x, int)]
    if only_ids:
        counts = Counter(only_ids)

        # get the sentence with the closest overlap
        closest, _ = counts.most_common(n=1)[0]

        rouge = Rouge()
        scores = rouge.get_scores(hyps=' '.join(document_sents_words[closest]), refs= ' '.join(sent_words))

        return scores[0]

    return None

def classify_result(result, doc_sents_words):
    sent_words = result['sent_words']
    state_vector = result['state_vector']

    only_ids = [x for x in state_vector if isinstance(x, int)]
    counts = Counter(only_ids)

    n_distinct_sents = len(counts)

    # totally new sentence
    if n_distinct_sents == 0:
        return {'type':' NEW'}
    # single sentence that was edited
    elif n_distinct_sents == 1:
        # get the sentence and compute the diff
        s_id = list(counts.keys())[0]
        candidate_words = doc_sents_words[s_id]
        candidate_words = [word for word in candidate_words if word is not '']
        # difflib's representation of deletions, insertions etc.
        opcodes = get_opcodes(sent_words, candidate_words)
        analysis = analyze_diff_operations(opcodes, sent_words, candidate_words)
        analysis['sent_ids'] = [s_id]
        return analysis

    # merger of multiple sentences
    else:
        # break this down further based on the share of each
        distribution = []
        total = sum(counts.values())
        for key in counts.keys():
            distribution.append((counts[key]) / float(total))
        analysis = {'type': 'MERGER', 'n_distinct_sents' : n_distinct_sents, 'distribution' : distribution,
                    'sent_ids': list(counts.keys())}
        return analysis


text = 'This sentence appears verbatim in the document. Stevie, 47, a resident of Tulsa, OK, was found to be a drug dealer. He was arrested in 2017.'

test_cases = {'This sentence appears verbatim in the document.': 'VERBATIM',
              'Stevie, 47, was arrested in 2017 and appears verbatim in the document.': 'MERGER',
              'Stevie was found to be a drug dealer.': 'DELETION',
              'Stevie, 47 was identified as a drug dealer.': 'REPLACEMENT',
              'Stevie, 47, a man from Tulsa, OK, was found to be a drug dealer.': 'INSERTION'}


text_sents = text.split('. ')
for sentence in test_cases.keys():
    print(analyze_sentence(sentence, text_sents))
    print(test_cases[sentence])


