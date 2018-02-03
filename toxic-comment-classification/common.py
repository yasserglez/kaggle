import os
import logging

import pandas as pd


logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)


DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')


LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def params_str(params):
    sorted_params = sorted(params.items(), key=lambda x: x[0])
    s = '_'.join(['{}={}'.format(k.replace('_', '-'), v) for k, v in sorted_params])
    return s


def load_data(mode, random_seed, file_name):
    path_parts = [DATA_DIR, mode]
    if mode == 'cross_validation':
        path_parts.append(str(random_seed))
    path_parts.append(file_name)
    columns = ['id']
    if file_name == 'train.csv':
        columns.extend(LABELS)
    df = pd.read_csv(os.path.join(*path_parts), usecols=columns)
    return df


def split_data(df, test_size, random_state):
    test_df = df.groupby(LABELS) \
        .apply(lambda x: x.sample(frac=test_size, random_state=random_state))
    train_df = df[~df['id'].isin(test_df['id'])]
    return train_df, test_df
