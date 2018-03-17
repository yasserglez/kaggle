import os
import logging

import numpy as np
from sklearn.model_selection import StratifiedKFold
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
    format_value = lambda v: '-'.join(sorted(v)) if isinstance(v, list) else v
    s = '_'.join(['{}={}'.format(k.replace('_', '-'), format_value(v)) for k, v in sorted_params])
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


def load_data(dataset):
    path = os.path.join(DATA_DIR, f'{dataset}.csv')
    cols = ['id'] + (LABELS if dataset == 'train' else []) + ['comment_text']
    df = pd.read_csv(path, usecols=cols)
    return df


def stratified_kfold(df, random_seed, k=10):
    # Assign a unique value to each label combination
    y = np.sum(df[LABELS].values * (2 ** np.arange(len(LABELS))), axis=1)
    X = np.zeros_like(y)  # Create a dummy X
    kfold = StratifiedKFold(n_splits=k, random_state=random_seed)
    for fold_num, (train_indices, val_indices) in enumerate(kfold.split(X, y), start=1):
        train_ids = set(df['id'].values[train_indices])
        val_ids = set(df['id'].values[val_indices])
        yield fold_num, train_ids, val_ids
