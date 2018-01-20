import os
import argparse

import pandas as pd
from numpy.random import RandomState

import common


def cv_split(random_seed):
    random_state = RandomState(random_seed)
    data = pd.read_csv(os.path.join(common.DATA_DIR, 'submission/train.csv'))
    train_data, test_data = common.split_data(data, test_size=0.2, random_state=random_state)

    cv_dir = os.path.join(common.DATA_DIR, 'cross_validation', str(random_seed))
    if not os.path.isdir(cv_dir):
        os.makedirs(cv_dir)
    train_data.to_csv(os.path.join(cv_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(cv_dir, 'test.csv'), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', type=int, required=True)
    args = parser.parse_args()
    cv_split(args.random_seed)


if __name__ == '__main__':
    main()
