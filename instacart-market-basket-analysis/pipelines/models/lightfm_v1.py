import pprint
import tempfile
import zipfile

import luigi
import ujson
import numpy as np
from numpy.random import RandomState
from scipy.sparse import dok_matrix
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from lightfm import LightFM
import pandas as pd

from ..models import FitModel, PredictModel
from ..clean_data import Products


class LightFMv1(object):

    loss = luigi.ChoiceParameter(choices=['logistic', 'bpr', 'warp'], default='logistic')
    no_components = luigi.IntParameter(default=10)
    max_sampled = luigi.IntParameter(default=100)

    random_seed = luigi.IntParameter(default=3996193, significant=False)
    epochs = luigi.IntParameter(default=100, significant=False)
    num_threads = luigi.IntParameter(default=8, significant=False)

    @property
    def model_name(self):
        params = [self.loss, self.no_components]
        if self.loss == 'warp':
            params.append(self.max_sampled)
        model_name = 'lightfm_v1_{}'.format('_'.join(str(p) for p in params))
        return model_name

    def _generate_row(self, last_order, prior_orders):
        # Collect the indices of the previously ordered products
        previously_ordered = set()
        for order in prior_orders:
            for product in order['products']:
                previously_ordered.add(product['product_id'] - 1)
        # Collect the indices of the reordered products
        reordered = set()
        for product in last_order['products']:
            if product['reordered']:
                reordered.add(product['product_id'] - 1)
        return last_order['order_id'], previously_ordered, reordered

    def _generate_rows(self, user_data, training):
        # if training:
        #     # Create new "users" from prior orders
        #     for k in range(1, len(user_data['prior_orders'])):
        #         last_order = user_data['prior_orders'][k]
        #         prior_orders = user_data['prior_orders'][:k]
        #         yield self._generate_row(last_order, prior_orders)
        yield self._generate_row(user_data['last_order'], user_data['prior_orders'])

    def _generate_matrices(self, orders_path, training):
        num_products = Products.count()
        order_ids, previously_ordered_sets, reordered_sets = [], [], []

        # Collect the data for the sparse matrices
        with open(orders_path) as f:
            for line in f:
                user_data = ujson.loads(line)
                for order_id, previously_ordered, reordered in self._generate_rows(user_data, training):
                    order_ids.append(order_id)
                    previously_ordered_sets.append(previously_ordered)
                    reordered_sets.append(reordered)

        # Populate the sparse matrices
        user_features_matrix = dok_matrix((len(order_ids), num_products), np.float32)
        interactions_matrix = dok_matrix((len(order_ids), num_products), np.float32)
        for i in range(len(order_ids)):
            for j in previously_ordered_sets[i]:
                user_features_matrix[i, j] = 1
                if j in reordered_sets[i]:
                    # Previously ordered and reordered -> positive interaction
                    interactions_matrix[i, j] = 1
                else:
                    # Previously ordered but did not reorder -> negative interaction
                    if self.loss == 'logistic':
                        # LightFM only supports negative interactions with the logistic loss
                        interactions_matrix[i, j] = -1
        user_features_matrix = user_features_matrix.tocsr()
        interactions_matrix = interactions_matrix.tocoo()

        return order_ids, user_features_matrix, interactions_matrix


class FitLightFMv1(LightFMv1, FitModel):

    def _fit_ranking_model(self, user_features, interactions):
        model = LightFMv1(no_components=self.no_components,
                          loss=self.loss,
                          max_sampled=self.max_sampled,
                          random_state=self.random)
        model.fit(interactions, user_features=user_features, epochs=self.epochs,
                  num_threads=self.num_threads, verbose=True)
        return model

    def _f1_score(self, y_true, y_pred):
        precision = len(y_true & y_pred) / len(y_pred)
        recall = len(y_true & y_pred) / len(y_true)
        f1 = 2.0 * (precision * recall) / (precision + recall) if precision or recall else 0.0
        return f1

    def _fit_threshold_model(self, user_features, interactions, ranking_model):
        interactions = interactions.tocsr()

        # Prepare the training data: find the score threshold that produces the maximum F1 value for each user
        y_true = []
        for i in range(user_features.shape[0]):
            # Calculate the score for each previously ordered product
            _, product_indices = user_features[i, :].nonzero()
            scores = ranking_model.predict(i, product_indices, user_features=user_features, num_threads=self.num_threads)
            prediction_scores = sorted(zip(product_indices, scores), key=lambda t: t[1], reverse=True)

            # Collect the correct predictions
            _, J = interactions[i, :].nonzero()
            reordered = {j for j in J if interactions[i, j] == 1}
            if not reordered:
                reordered.add('None')

            # Determine the optimal threshold for the user
            best_threshold = prediction_scores[0][1] + np.finfo(np.float).eps
            best_f1 = self._f1_score(reordered, {'None'})
            current_predictions = set()
            for product_index, current_threshold in prediction_scores:
                current_predictions.add(product_index)
                current_f1 = self._f1_score(reordered, current_predictions)
                if current_f1 > best_f1:
                    best_threshold = current_threshold
                    best_f1 = current_f1
            y_true.append(best_threshold)

        # Use the LightFM user embedding as predictors
        biases, embeddings = ranking_model.get_user_representations(user_features)
        X = np.hstack((embeddings, biases.reshape(-1, 1)))

        # Train a regression model
        model = GradientBoostingRegressor(loss='ls', random_state=self.random)
        model_params = {
            'n_estimators': [500],
            'max_depth': [3, 5, 7],
        }
        grid_search = GridSearchCV(model, param_grid=model_params,
                                   scoring='neg_mean_squared_error', cv=10,
                                   n_jobs=self.num_threads, verbose=True)
        grid_search.fit(X, y_true)
        print(); pprint.pprint(grid_search.best_params_)
        threshold_model = grid_search.best_estimator_

        y_pred = threshold_model.predict(X)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        df['absolute_error'] = np.abs(df['y_true'] - df['y_pred'])
        print('\n', df.describe())
        print('\nr^2:', r2, '\nrmse:', rmse, '\n')

        return threshold_model

    def run(self):
        self.random = RandomState(self.random_seed)

        orders_path = self.requires()['orders'].output().path
        order_ids, user_features, interactions = self._generate_matrices(orders_path, training=True)

        ranking_model = self._fit_ranking_model(user_features, interactions)
        threshold_model = self._fit_threshold_model(user_features, interactions, ranking_model)

        with tempfile.NamedTemporaryFile() as ranking_file:
            joblib.dump(ranking_model, ranking_file.name)
            with tempfile.NamedTemporaryFile() as threshold_file:
                joblib.dump(threshold_model, threshold_file.name)
                with zipfile.ZipFile(self.output().path, 'w') as zip:
                    zip.write(ranking_file.name, 'ranking_model')
                    zip.write(threshold_file.name, 'threshold_model')


class PredictLightFMv1(LightFMv1, PredictModel):

    def requires(self):
        req = super().requires()
        req['model'] = FitLightFMv1(
            mode=self.mode,
            loss=self.loss,
            no_components=self.no_components,
            max_sampled=self.max_sampled)
        return req

    def run(self):
        self.random = RandomState(self.random_seed)

        orders_path = self.requires()['orders'].output().path
        order_ids, user_features, _ = self._generate_matrices(orders_path, training=False)

        with zipfile.ZipFile(self.input()['model'].path) as zip:
            with tempfile.NamedTemporaryFile() as ranking_file:
                f = zip.open('ranking_model')
                ranking_file.write(f.read())
                ranking_file.flush()
                ranking_model = joblib.load(ranking_file.name)
                with tempfile.NamedTemporaryFile() as threshold_file:
                    f = zip.open('threshold_model')
                    threshold_file.write(f.read())
                    threshold_file.flush()
                    threshold_model = joblib.load(threshold_file.name)

        # Predict the thresholds
        biases, embeddings = ranking_model.get_user_representations(user_features)
        X = np.hstack((embeddings, biases.reshape(-1, 1)))
        thresholds = threshold_model.predict(X)

        # Compute the score for each previously ordered product
        predictions = {}
        for i in range(len(order_ids)):
            _, previously_ordered = user_features[i, :].nonzero()
            scores = ranking_model.predict(i, previously_ordered, user_features=user_features, num_threads=self.num_threads)
            df = pd.DataFrame({'product_id': previously_ordered + 1, 'score': scores})
            df = df[df.score >= thresholds[i]].sort_values('score', ascending=False)

            predictions[order_ids[i]] = []
            for row in df.itertuples(index=False):
                # ujson fails when it tries to serialize the numpy int values
                predictions[order_ids[i]].append(int(row.product_id))

        with self.output().open('w') as fd:
            ujson.dump(predictions, fd)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
