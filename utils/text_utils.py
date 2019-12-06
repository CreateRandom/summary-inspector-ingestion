from nltk import word_tokenize, ngrams


def get_novel_ngrams(summary, document, n=2, language='english'):
    if not isinstance(document, list):
        document = word_tokenize(document, language)
    doc_grams = set(ngrams(document, n=n))
    if not isinstance(summary, list):
        summary = word_tokenize(summary, language)
    sent_grams = set(ngrams(summary, n=n))
    new_grams = [x[0] for x in sent_grams if x not in doc_grams]
    return new_grams


def spacy_sent_tokenize(text, spacy_model):
    doc = spacy_model(text)
    return [sent.string.strip() for sent in doc.sents]

def wrap_in_tag(text, tag):
    return '<' + tag + '>' + text + '</' + tag + '>'