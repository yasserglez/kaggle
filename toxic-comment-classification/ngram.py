# Based on https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams/

import os
import sys
import pprint
import logging
from collections import defaultdict
from datetime import datetime

from unidecode import unidecode
import numpy as np
from numpy.random import RandomState
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

import common
import base


logger = logging.getLogger(__name__)


class NGram(base.BaseModel):

    def main(self):
        t_start = datetime.now()
        logger.info(' {} / {} '.format(self.name, self.random_seed).center(62, '='))
        logger.info('Hyperparameters:\n{}'.format(pprint.pformat(self.params)))
        if os.path.isfile(os.path.join(self.output_dir, 'test.csv')):
            logger.info('Output already exists - skipping')
            return

        # Initialize the random number generator
        self.random_state = RandomState(self.random_seed)
        np.random.seed(int.from_bytes(self.random_state.bytes(4), byteorder=sys.byteorder))

        train_df = common.load_data('train')
        train_df['comment_text'] = train_df['comment_text'].apply(unidecode)
        test_df = common.load_data('test')
        test_df['comment_text'] = test_df['comment_text'].apply(unidecode)

        vectorizer = self.build_vectorizer(train_df, test_df)

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
        X_train = vectorizer.transform(train_df['comment_text'])

        models = {}
        for label in common.LABELS:
            logger.info('Training the %s model', label)
            y_train = train_df[label].values
            model = LogisticRegression(
                solver='sag',
                penalty='l2',
                C=self.params['C'],
                tol=1e-8,
                max_iter=1000,
                random_state=self.random_state,
                verbose=1)
            model.fit(X_train, y_train)
            models[label] = model

        path = os.path.join(self.output_dir, f'fold{fold_num}.pickle')
        joblib.dump((vectorizer, models), path)
        return models

    def predict(self, models, vectorizer, df, output_path):
        X = vectorizer.transform(df['comment_text'])
        output = defaultdict(list)
        for label in common.LABELS:
            model = models[label]
            yhat = model.predict_proba(X)[:, 1]
            output[label].extend(yhat)
        predictions = pd.DataFrame.from_dict(output)
        predictions = predictions[common.LABELS]
        predictions.insert(0, 'id', df['id'].values)
        predictions.to_csv(output_path, index=False)

    def build_vectorizer(self, train_df, test_df):
        logger.info('Learning the vocabulary')

        vectorizer = TfidfVectorizer(
            strip_accents='unicode',
            analyzer=self.params['analyzer'],
            min_df=self.params['min_df'],
            ngram_range=(1, self.params['max_ngram']),
            max_features=self.params['max_features'],
            stop_words='english',
            sublinear_tf=True)

        train_text = train_df['comment_text']
        test_text = test_df['comment_text']
        all_text = pd.concat([train_text, test_text])
        vectorizer.fit(all_text)
        logger.info('The vocabulary has %s words (%s ignored as stopwords)', 
                    len(vectorizer.vocabulary_), len(vectorizer.stop_words_))

        return vectorizer


if __name__ == '__main__':
    params = {
        'analyzer': 'char',
        'min_df': 5,
        'max_ngram': 5,
        'max_features': 100000,
        'C': 1.0,
    }
    model = NGram(params, random_seed=base.RANDOM_SEED)
    model.main()

    params = {
        'analyzer': 'word',
        'min_df': 5,
        'max_ngram': 2,
        'max_features': 50000,
        'C': 1.0,
    }
    model = NGram(params, random_seed=base.RANDOM_SEED)
    model.main()
