import os
import re
import string
import logging
from collections import Counter
import multiprocessing as mp

import joblib
from unidecode import unidecode

import spacy
english = spacy.load('en')

import common


logger = logging.getLogger(__name__)


OUTPUT_DIR = os.path.join(common.OUTPUT_DIR, 'preprocessing')
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def load(params):
    """Load the preprocessed data.

    Parameters:
        vocab_size: maximum number of words to consider in the vocabulary.
        max_len: maximum number of words to consider per example.
    """
    param_names = ('vocab_size', 'max_len')
    params = {k: params[k] for k in param_names}
    output_file = os.path.join(OUTPUT_DIR, common.params_str(params))

    if os.path.isfile(output_file):
        logger.info(f'Loading {output_file[len(common.OUTPUT_DIR) + 1:]}')
        preprocessed_data = joblib.load(output_file)
        return preprocessed_data

    logger.info(f'Generating {output_file[len(common.OUTPUT_DIR) + 1:]}')

    preprocessed_data = {}
    raw_data = common.load_raw_data()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap_unordered(preprocess_row, raw_data.items(), chunksize=100)
        for k, (id_, words) in enumerate(results, start=1):
            preprocessed_data[id_] = words
            if k % 10000 == 0 or k == len(raw_data):
                logger.info('Preprocessed {:,} / {:,} comments'.format(k, len(raw_data)))

    logger.info('Selecting the vocabulary')
    counter = Counter()
    for id_, words in preprocessed_data.items():
        counter.update(set(words))
    vocab = set(word for word, count in counter.most_common(params['vocab_size']))

    logger.info('Replacing out-of-vocabulary words with <UNK>')
    for id_, words in preprocessed_data.items():
        new_words = [w if w in vocab else '<UNK>' for w in words]
        preprocessed_data[id_] = ' '.join(new_words[:params['max_len']])
    joblib.dump(preprocessed_data, output_file)

    return preprocessed_data


def preprocess_row(row):
    return row[0], preprocess(row[1])


def preprocess(text):
    # Strip whitespaces, double quotes in the CSV, Wikipedia syntax for indentation
    text = text.strip().strip('"').lstrip(':')

    # Delete all text for some comments
    if text.startswith('REDIRECT'):
        return ['<URL>']
    if 'cellpadding' in text and 'cellspacing' in text:
        return ['<URL>']

    # Ignore everything enclosed in {}
    text = re.sub(r'\{[^{}]+\}', '', text)

    # Lowercase
    text = text.lower()

    # Convert to ASCII (unidecode does some clever transliterations)
    text = unidecode(text)

    # Wikipedia namespaces that will be replaced below by <URL>
    text = re.sub(r'image:.+\.(jpg|jpeg|gif)', ' http://image.jpg ', text)
    text = re.sub(r'[a-z_]+:[^\s]+', ' http://namespace.com ', text)

    # Replace sequences of whitespace characters with a single space
    text = re.sub(r'\s{2,}', ' ', text)

    # Collapse all characters repeated 4+ times to 3 occurrences
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
    # and specific characters to 1 occurrence
    text = re.sub(r'(["])+', r'\1', text)

    # Remove IP addresses
    text = ip_re.sub('', text)

    # Replace specific words
    for old_re, new in word_replace.items():
        text = old_re.sub(new, text)

    words = []
    for token in english.tokenizer(text):
        word = token.text

        if word.isspace():
            continue

        if word.isnumeric() or number_re.match(word):
            word = '<NUM>'
        elif url_re.match(word):
            word = '<URL>'

        words.append(word)

    if not words:
        words.append('<UNK>')

    return words


word_replace = {
    re.compile('f[' + string.punctuation + ']+ck'): 'fuck',
    re.compile('fu[' + string.punctuation + ']+k'): 'fuck',
    re.compile('f[' + string.punctuation + ']{2,}k'): 'fuck',
    re.compile('fuk{2,}'): 'fuck',
    re.compile('fuck{2,}'): 'fuck',
    re.compile('f[' + string.punctuation + ']+ggot'): 'faggot',
    re.compile('fagg[' + string.punctuation + ']+t'): 'faggot',
    re.compile('s[' + string.punctuation + ']+it'): 'shit',
    re.compile('sh[' + string.punctuation + '1' + ']+t'): 'shit',
    re.compile('s[' + string.punctuation + ']{2,}t'): 'shit',
    re.compile('gayreek'): 'gay greek',
}

allowed_chars = set(string.ascii_lowercase + '."/!,;\'-')


# https://www.regextester.com/93652
url_re = re.compile(r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?')

# https://www.safaribooksonline.com/library/view/regular-expressions-cookbook/9781449327453/ch06s11.html
number_re = re.compile(r'[0-9]{1,3}(,[0-9]{3})*(\.[0-9]+)?\b|\.[0-9]+')

ip_re = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
