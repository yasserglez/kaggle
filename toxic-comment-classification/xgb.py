import os
import sys
import pprint
import logging
import multiprocessing as mp
from collections import defaultdict

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
    try:
        y_predict = estimator.predict_proba(X, ntree_limit=estimator.best_ntree_limit)
    except AttributeError:
        y_predict = estimator.predict_proba(X)
    return roc_auc_score(y, y_predict[:, 1])


class XGB(base.BaseModel):

    def __init__(self, name, params, random_seed):
        super().__init__(name, params, random_seed, debug=False)

    def main(self):
        logger.info(' {} / {} '.format(self.name, self.random_seed).center(62, '='))
        logger.info('Hyperparameters:\n{}'.format(pprint.pformat(self.params)))
        if os.path.isfile(self.validation_output):
            logger.info('Output already exists - skipping')
        else:
            self.random_state = RandomState(self.random_seed)
            np.random.seed(int.from_bytes(self.random_state.bytes(4), byteorder=sys.byteorder))
            preprocessed_data = preprocessing.load(self.params)
            self.train(preprocessed_data)
            self.predict(preprocessed_data)

    def train(self, preprocessed_data):
        vectorizer = self.build_vectorizer(preprocessed_data)

        train_df = common.load_data(self.random_seed, 'train')
        train_df['text'] = train_df['id'].map(preprocessed_data)
        X_train = vectorizer.transform(train_df['text'])

        val_df = common.load_data(self.random_seed, 'validation')
        val_df['text'] = val_df['id'].map(preprocessed_data)
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

        joblib.dump((vectorizer, models), self.model_output)

    def predict(self, preprocessed_data):
        vectorizer, models = joblib.load(self.model_output)
        for dataset in ['train', 'validation', 'test']:
            csv_file = (self.train_output if dataset == 'train' else
                        self.validation_output if dataset == 'validation' else
                        self.test_output)
            logger.info('Generating {} predictions'.format(dataset))
            df = common.load_data(self.random_seed, dataset)
            df['text'] = df['id'].map(preprocessed_data)
            X = vectorizer.transform(df['text'])
            output = defaultdict(list)
            for label in common.LABELS:
                model = models[label]
                yhat = model.predict_proba(X, ntree_limit=model.best_ntree_limit)[:, 1]
                output[label].extend(yhat)
            predictions = pd.DataFrame.from_dict(output)
            predictions = predictions[common.LABELS]
            predictions.insert(0, 'id', df['id'].values)
            predictions.to_csv(csv_file, index=False)

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
    model = XGB('xgb', params, random_seed=42)
    model.main()
