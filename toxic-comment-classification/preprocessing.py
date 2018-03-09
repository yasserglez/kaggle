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
    vocab = set(word for word, count in counter.most_common(params['vocab_size'] - 1))

    logger.info('Replacing out-of-vocabulary words with <UNK>')
    for id_, words in preprocessed_data.items():
        new_words = [w if w in vocab else '<UNK>' for w in words]
        preprocessed_data[id_] = ' '.join(new_words[:params['max_len']])
    joblib.dump(preprocessed_data, output_file)

    return preprocessed_data


def preprocess_row(row):
    return row[0], preprocess(row[1])


def preprocess(text):

    # Convert to ASCII (unidecode does some clever transliterations)
    text = unidecode(text)

    # Strip whitespaces, double quotes in the CSV, Wikipedia syntax for indentation
    text = text.strip().strip('"').lstrip(':')

    # Delete all text for some comments
    if text.startswith('REDIRECT'):
        return ['<URL>']
    if 'cellpadding' in text and 'cellspacing' in text:
        return ['<URL>']

    # Lowercase
    text = text.lower()

    # Ignore everything enclosed in {}
    if text.startswith('{{unblock|'):
        text = text[10:].rstrip('{}')
    else:
        text = re.sub(r'\{[^{}]+\}', '', text)

    # Wikipedia specific syntax
    text = re.sub(r'image:.+\.(jpg|jpeg|gif)', ' http://image.jpg ', text)
    text = re.sub(r'[a-z_]+:[^\s]+', ' http://namespace.com ', text)
    if text.endswith('talk/email'):
        text = text[:-10].rstrip(' -')

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

        # Ignore spaces
        if word.isspace():
            continue

        # Use a manually filtered down list of short tokens
        if len(word) <= 3 and word not in short_words:
            word = '<UNK>'

        # Normalize haha and lol
        for x in 'aeiou':
            for k in range(5, 1, -1):
                hx = ('h' + x) * k
                if word.startswith(hx):
                    word = hx
                    break
        for k in range(5, 1, -1):
            lol = 'lo' * k
            if word.startswith(lol):
                word = lol
                break

        if word.isnumeric() or number_re.match(word):
            word = '<NUM>'

        if '@' in word:
            words.append('<URL>')
        elif url_re.match(word):
            if re.match(r'[^.]+\.[^.]+', word):
                word_parts = word.split('.')
                if len(word_parts[-1]) > 3:
                    # Separate words joined by a period
                    words.extend(word_parts)
                else:
                    words.append('<URL>')
            else:
                words.append('<URL>')
        else:
            # All the other words are added here
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
    re.compile('f[' + string.punctuation + ']+ing'): 'fucking',
    re.compile('f[' + string.punctuation + ']+ggot'): 'faggot',
    re.compile('fagg[' + string.punctuation + ']+t'): 'faggot',
    re.compile('s[' + string.punctuation + ']+it'): 'shit',
    re.compile('sh[' + string.punctuation + '1' + ']+t'): 'shit',
    re.compile('s[' + string.punctuation + ']{2,}t'): 'shit',
    re.compile('s[' + string.punctuation + ']+ck'): 'suck',
    re.compile('s[' + string.punctuation + ']+uck'): 'suck',
    re.compile('su[' + string.punctuation + ']+ck'): 'suck',
    re.compile('suc[' + string.punctuation + ']+k'): 'suck',
    re.compile('c[' + string.punctuation + ']+ck'): 'cock',
    re.compile('c[' + string.punctuation + ']+ock'): 'cock',
    re.compile('co[' + string.punctuation + ']+ck'): 'cock',
    re.compile('coc[' + string.punctuation + ']+k'): 'cock',
}


# https://www.regextester.com/93652
url_re = re.compile(r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?')

# https://www.safaribooksonline.com/library/view/regular-expressions-cookbook/9781449327453/ch06s11.html
number_re = re.compile(r'[0-9]{1,3}(,[0-9]{3})*(\.[0-9]+)?\b|\.[0-9]+')

ip_re = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')


short_words = {
    '!', '"', '$', '%', '&', "'", "'d", "'em", "'ll", "'m", "'re", "'s",
    "'ve", '(', ')', ',', '-', '--', '.', '...', '/', '1st', '2nd',
    '3rd', '4th', '5th', '6th', '7th', '8th', '9th', ':', ':*', ';',
    ';)', ';-)', '<3', '=)', '?', '[', ']', 'a', 'abc', 'act', 'ad',
    'add', 'ads', 'afd', 'age', 'ago', 'ah', 'ahh', 'ai', 'aid', 'aim',
    'air', 'aka', 'ali', 'all', 'alt', 'am', 'an', 'and', 'ani', 'ann',
    'any', 'aol', 'apr', 'are', 'arm', 'art', 'as', 'ask', 'ass', 'at',
    'ate', 'aug', 'axe', 'bad', 'bag', 'ban', 'bar', 'bat', 'bay', 'bbc',
    'be', 'bed', 'beg', 'ben', 'bet', 'big', 'bin', 'bio', 'bit', 'bob',
    'bot', 'bow', 'box', 'boy', 'bro', 'btw', 'bug', 'bum', 'bus', 'but',
    'buy', 'by', 'bye', 'can', 'cap', 'car', 'cat', 'cbs', 'ceo', 'cia',
    'cnn', 'coi', 'com', 'con', 'cop', 'cos', 'cow', 'cry', 'csd', 'cum',
    'cup', 'cut', 'cuz', 'dab', 'dad', 'dam', 'dan', 'dat', 'day', 'de',
    'dec', 'del', 'den', 'der', 'did', 'die', 'dig', 'dis', 'dna', 'do',
    'doc', 'dog', 'don', 'dot', 'dr', 'dry', 'due', 'duh', 'dvd', 'dyk',
    'ear', 'eat', 'eg', 'egg', 'ego', 'eh', 'emo', 'en', 'end', 'era',
    'err', 'esp', 'est', 'et', 'etc', 'eye', 'fac', 'fag', 'fan', 'faq',
    'far', 'fat', 'fbi', 'feb', 'fed', 'few', 'fit', 'fix', 'fly', 'for',
    'fox', 'fun', 'fur', 'fyi', 'gan', 'gap', 'gas', 'gay', 'get', 'gnu',
    'go', 'god', 'gon', 'got', 'gun', 'guy', 'ha', 'had', 'has', 'hat',
    'he', 'heh', 'her', 'hey', 'hi', 'him', 'hip', 'his', 'hit', 'hiv',
    'hmm', 'hoe', 'hop', 'hot', 'how', 'hub', 'huh', 'i', "i'm", 'ian',
    'ice', 'ie', 'if', 'iii', 'ill', 'imo', 'in', 'inc', 'ip', 'ipa',
    'ips', 'irc', 'is', 'isp', 'it', 'its', 'jan', 'jay', 'jet', 'jew',
    'jim', 'job', 'joe', 'joy', 'jun', 'ken', 'key', 'kid', 'kim', 'kkk',
    'lab', 'law', 'lay', 'led', 'lee', 'leg', 'let', 'lie', 'lil', 'log',
    'lol', 'los', 'lot', 'low', 'mac', 'mad', 'man', 'map', 'mar', 'max',
    'may', 'me', 'mel', 'men', 'met', 'mid', 'mis', 'mit', 'mix', 'mmm',
    'mob', 'mod', 'mom', 'mos', 'mr', 'mrs', 'mtv', 'mud', 'mum', 'my',
    "n't", 'nah', 'neo', 'net', 'new', 'nfl', 'no', 'nom', 'non', 'nor',
    'not', 'nov', 'now', 'nt', 'nut', 'oct', 'odd', 'of', 'off', 'oh',
    'oil', 'ok', 'old', 'omg', 'on', 'one', 'opt', 'or', 'org', 'our',
    'out', 'owe', 'own', 'pal', 'pan', 'par', 'pat', 'pay', 'pdf', 'pen',
    'per', 'pet', 'phd', 'pic', 'pie', 'pig', 'pin', 'pit', 'pls', 'plz',
    'poo', 'pop', 'pot', 'pov', 'ppl', 'pre', 'pro', 'ps', 'pun', 'put',
    'que', 'quo', 'ran', 'rap', 'rat', 'raw', 'ray', 'red', 'ref', 'rev',
    'rfa', 'rfc', 'rid', 'rip', 'rob', 'ron', 'rot', 'row', 'roy', 'run',
    'sad', 'sam', 'san', 'sat', 'saw', 'say', 'sea', 'see', 'sep', 'set',
    'sex', 'she', 'sic', 'sig', 'sin', 'sir', 'sit', 'six', 'sky', 'so',
    'son', 'spi', 'sub', 'sue', 'sum', 'sun', 'tab', 'tad', 'tag', 'tax',
    'tea', 'ted', 'teh', 'ten', 'tfd', 'the', 'tho', 'thx', 'tie', 'tim',
    'tip', 'to', 'tom', 'ton', 'too', 'top', 'toy', 'try', 'tv', 'two',
    'u.s', 'ugh', 'uk', 'umm', 'up', 'ups', 'ur', 'url', 'us', 'usa',
    'use', 'utc', 'van', 'vfd', 'via', 'vol', 'von', 'vs', 'vs.', 'war',
    'was', 'wat', 'way', 'we', 'web', 'wee', 'wet', 'who', 'why', 'win',
    'wit', 'won', 'wow', 'wtf', 'ww2', 'wwe', 'xxx', 'yay', 'yea', 'yep',
    'yes', 'yet', 'yo', 'you', 'yup',
}
