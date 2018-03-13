import os
import sys
import pprint
import logging
import multiprocessing as mp
from collections import defaultdict
from datetime import datetime

import numpy as np
from numpy.random import RandomState
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib

import common
import base
import preprocessing


logger = logging.getLogger(__name__)


def scoring(estimator, X, y):
    best_n_tree_limit = getattr(estimator, 'best_n_tree_limit', None)
    y_predict = estimator.predict_proba(X, ntree_limit=best_ntree_limit)
    return roc_auc_score(y, y_predict[:, 1])


class XGB(base.BaseModel):

    def __init__(self, name, params, random_seed):
        super().__init__(name, params, random_seed)

    def main(self):
        t_start = datetime.now()
        logger.info(' {} / {} '.format(self.name, self.random_seed).center(62, '='))
        logger.info('Hyperparameters:\n{}'.format(pprint.pformat(self.params)))
        if os.path.isfile(os.path.join(self.output_dir, 'test.csv')):
            logger.info('Output already exists - skipping')

        # Initialize the random number generator
        self.random_state = RandomState(self.random_seed)
        np.random.seed(int.from_bytes(self.random_state.bytes(4), byteorder=sys.byteorder))

        preprocessed_data = preprocessing.load(self.params)
        vectorizer = self.build_vectorizer(preprocessed_data)

        train_df = common.load_data('train')
        train_df['text'] = train_df['id'].map(preprocessed_data)
        test_df = common.load_data('test')
        test_df['text'] = test_df['id'].map(preprocessed_data)

        folds = common.stratified_kfold(train_df, random_seed=self.random_seed)
        for fold_num, train_ids, val_ids in folds:
            logger.info(f'Fold #{fold_num}')

            fold_train_df = train_df[train_df['id'].isin(train_ids)]
            fold_val_df = train_df[train_df['id'].isin(val_ids)]
            models = self.train(fold_num, vectorizer, fold_train_df, fold_val_df)

            logger.info('Generating the out-of-fold predictions')
            path = os.path.join(self.output_dir, f'fold{fold_num}_validation.csv')
            self.predict(models, vectorizer, fold_val_df, path)

            logger.info('Generating the test predictions')
            path = os.path.join(self.output_dir, f'fold{fold_num}_test.csv')
            self.predict(models, vectorizer, test_df, path)

        logger.info('Combining the out-of-fold predictions')
        df_parts = []
        for fold_num in range(1, 11):
            path = os.path.join(self.output_dir, f'fold{fold_num}_validation.csv')
            df_part = pd.read_csv(path, usecols=['id'] + common.LABELS)
            df_parts.append(df_part)
        train_pred = pd.concat(df_parts)
        path = os.path.join(self.output_dir, 'train.csv')
        train_pred.to_csv(path, index=False)

        logger.info('Averaging the test predictions')
        df_parts = []
        for fold_num in range(1, 11):
            path = os.path.join(self.output_dir, f'fold{fold_num}_test.csv')
            df_part = pd.read_csv(path, usecols=['id'] + common.LABELS)
            df_parts.append(df_part)
        test_pred = pd.concat(df_parts).groupby('id', as_index=False).mean()
        path = os.path.join(self.output_dir, 'test.csv')
        test_pred.to_csv(path, index=False)

        logger.info('Total elapsed time - {}'.format(datetime.now() - t_start))

    def train(self, fold_num, vectorizer, train_df, val_df):
        X_train = vectorizer.transform(train_df['text'])
        X_val = vectorizer.transform(val_df['text'])

        models = {}
        for label in common.LABELS:
            logger.info('Training the %s model', label)
            y_train, y_val = train_df[label].values, val_df[label].values
            scale_pos_weight = (1 - y_train).sum() / y_train.sum()

            model = xgb.XGBClassifier(
                    n_estimators=10000,  # determined by early stopping
                    max_depth=self.params['max_depth'],
                    objective='binary:logistic',
                    learning_rate=self.params['learning_rate'],
                    scale_pos_weight=scale_pos_weight,
                    min_child_weight=scale_pos_weight,
                    random_state=self.random_seed,
                    n_jobs=mp.cpu_count())

            model.fit(X_train, y_train,
                      eval_metric='auc', eval_set=[(X_val, y_val)],
                      early_stopping_rounds=10, verbose=True)

            models[label] = model

        path = os.path.join(self.output_dir, f'fold{fold_num}.pickle')
        joblib.dump((vectorizer, models), path)
        return models

    def predict(self, models, vectorizer, df, output_path):
        X = vectorizer.transform(df['text'])
        output = defaultdict(list)
        for label in common.LABELS:
            model = models[label]
            yhat = model.predict_proba(X, ntree_limit=model.best_ntree_limit)[:, 1]
            output[label].extend(yhat)
        predictions = pd.DataFrame.from_dict(output)
        predictions = predictions[common.LABELS]
        predictions.insert(0, 'id', df['id'].values)
        predictions.to_csv(output_path, index=False)

    def build_vectorizer(self, preprocessed_data):
        logger.info('Learning the vocabulary')
        vectorizer = TfidfVectorizer(min_df=self.params['min_df'])
        vectorizer.fit(preprocessed_data.values())
        logger.info('The vocabulary has %s words (%s ignored as stopwords)', 
                    len(vectorizer.vocabulary_), len(vectorizer.stop_words_))
        return vectorizer


if __name__ == '__main__':
    params = {
        'vocab_size': 300000,
        'max_len': 1000,
        'min_df': 5,
        'learning_rate': 0.1,
        'max_depth': 6,
    }
    model = XGB('xgb', params, random_seed=base.RANDOM_SEED)
    model.main()
