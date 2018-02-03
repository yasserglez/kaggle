import os
import re
import logging
from collections import Counter
import multiprocessing as mp

import joblib
import pandas as pd

import spacy
english = spacy.load('en')

import common


logger = logging.getLogger(__name__)


BASE_DIR = os.path.join(common.DATA_DIR, 'preprocessing')
if not os.path.isdir(BASE_DIR):
    os.makedirs(BASE_DIR)


def load(params):
    """Load the preprocessed data.

    Parameters:
        vocab_size: maximum number of words to consider in the vocabulary.
        max_len: maximum number of words to consider per example.
    """
    param_names = ('vocab_size', 'max_len')
    params = {k: params[k] for k in param_names}
    output_file = os.path.join(BASE_DIR, common.params_str(params))

    if os.path.isfile(output_file):
        logger.info(f'Loading {output_file[len(common.DATA_DIR) + 1:]}')
        preprocessed_data = joblib.load(output_file)
        return preprocessed_data

    logger.info(f'Generating {output_file[len(common.DATA_DIR) + 1:]}')

    preprocessed_data = {}
    raw_data = load_raw_data()
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


def load_raw_data():
    df_parts = []
    for csv_file in ['train.csv', 'test.csv']:
        csv_path = os.path.join(common.DATA_DIR, 'submission', csv_file)
        df_part = pd.read_csv(csv_path, usecols=['id', 'comment_text'])
        df_parts.append(df_part)
    df = pd.concat(df_parts)
    df['comment_text'].fillna('', inplace=True)
    raw_data = {row[0]: row[1] for row in df.itertuples(index=False)}
    return raw_data


def preprocess_row(row):
    return row[0], preprocess(row[1])


def preprocess(text):
    # Wikipedia syntax for indentation, wrong quotes in CSV?
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

    # Wikipedia namespaces that will be replaced by <URL>
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

    # Replace specific characters
    replace_map = {
        '…': '...',
        '—': '-',
        '“': '"',
        '”': '"',
        '=': '',
    }
    for old, new in replace_map.items():
        text = text.replace(old, new)

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
        words.append('<URL>')

    return words


# https://www.regextester.com/93652
url_re = re.compile(r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?')

# https://www.safaribooksonline.com/library/view/regular-expressions-cookbook/9781449327453/ch06s11.html
number_re = re.compile(r'[0-9]{1,3}(,[0-9]{3})*(\.[0-9]+)?\b|\.[0-9]+')

ip_re = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
