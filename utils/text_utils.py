import difflib

from nltk import word_tokenize, ngrams


def spacy_sent_tokenize(text, spacy_model):
    doc = spacy_model(text)
    return [sent.string.strip() for sent in doc.sents]

def wrap_in_tag(text, tag):
    return '<' + tag + '>' + text + '</' + tag + '>'


def generate_overlapping_sections(text_a, text_b):
    doc_sents = text_a.split('\n\n')
    # TODO finetune this
    text_a = text_a.replace('\n', ' ')
    # assume this is a str that contains sentences split by newlines
    sents_b = text_b.split('\n')

    text_a = text_a.split(' ')

    sm = difflib.SequenceMatcher()
    string_matches = []

    for sent_b in sents_b:

        string_matches_local = []
        if sent_b is '':
            continue

        sent_b = sent_b.split(' ')
        sm.set_seqs(text_a, sent_b)

        matches = sm.get_matching_blocks()
        for match in matches:
            a, b, n = match
            if n == 0:
                continue
            words_a = text_a[a:a + n]
            concatenated = ' '.join(words_a)
            if concatenated is '':
                continue
            string_matches_local.append(concatenated)
        string_matches.append(string_matches_local)

    return string_matches