import os
import logging

from numpy.random import RandomState
import pandas as pd


logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(SRC_DIR, 'data')
OUTPUT_DIR = os.path.join(SRC_DIR, 'output')


LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def params_str(params):
    sorted_params = sorted(params.items(), key=lambda x: x[0])
    s = '_'.join(['{}={}'.format(k.replace('_', '-'), v) for k, v in sorted_params])
    return s


def load_raw_data():
    df_parts = []
    for csv_file in ['train.csv', 'test.csv']:
        csv_path = os.path.join(DATA_DIR, csv_file)
        df_part = pd.read_csv(csv_path, usecols=['id', 'comment_text'])
        df_parts.append(df_part)
    df = pd.concat(df_parts)
    df['comment_text'].fillna('', inplace=True)
    raw_data = {row[0]: row[1] for row in df.itertuples(index=False)}
    return raw_data


def load_data(random_seed, dataset):
    if dataset in {'train', 'validation'}:
        random_state = RandomState(random_seed)
        data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), usecols=['id'] + LABELS)
        train_data, validation_data = split_data(data, test_size=0.25, random_state=random_state)
        return train_data if dataset == 'train' else validation_data
    elif dataset == 'test':
        test_data = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), usecols=['id'])
        return test_data


def split_data(df, test_size, random_state):
    test_df = df.groupby(LABELS, as_index=False) \
        .apply(lambda x: x.sample(frac=test_size, random_state=random_state))
    train_df = df[~df['id'].isin(test_df['id'])]
    return train_df, test_df
