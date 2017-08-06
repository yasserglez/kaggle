import pprint

import luigi
import ujson
import numpy as np
from numpy.random import RandomState
from scipy.sparse import dok_matrix
import pandas as pd
from sklearn.externals import joblib
from lightfm import LightFM

from ..models import FitModel, PredictModel
from ..clean_data import Products


class _LightFM(object):

    loss = luigi.ChoiceParameter(choices=['logistic', 'bpr', 'warp'], default='logistic')
    no_components = luigi.IntParameter(default=100)
    max_prior_orders = luigi.IntParameter(default=5)
    max_sampled = luigi.IntParameter(default=100)

    random_seed = luigi.IntParameter(default=3996193, significant=False)
    epochs = luigi.IntParameter(default=100, significant=False)
    num_threads = luigi.IntParameter(default=16, significant=False)

    num_products = Products.count()

    @property
    def model_name(self):
        params = [self.loss, self.no_components, self.max_prior_orders]
        if self.loss == 'warp':
            params.append(self.max_sampled)
        model_name = 'lightfm_v2_{}'.format('_'.join(str(p) for p in params))
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

    def _generate_rows(self, user_data, max_prior_orders):
        yield self._generate_row(user_data['last_order'], user_data['prior_orders'])
        max_prior_orders -= 1
        if max_prior_orders > 0:
            for k in range(len(user_data['prior_orders']) - 1, 0, -1):
                last_order = user_data['prior_orders'][k]
                prior_orders = user_data['prior_orders'][:k]
                yield self._generate_row(last_order, prior_orders)
                max_prior_orders -= 1
                if max_prior_orders == 0:
                    break

    def _generate_matrices(self, orders_path, max_prior_orders):
        order_ids, previously_ordered_sets, reordered_sets = [], [], []

        # Collect the data for the sparse matrices
        with open(orders_path) as f:
            for line in f:
                user_data = ujson.loads(line)
                for order_id, previously_ordered, reordered in self._generate_rows(user_data, max_prior_orders):
                    order_ids.append(order_id)
                    previously_ordered_sets.append(previously_ordered)
                    reordered_sets.append(reordered)

        # Populate the sparse matrices
        user_features_matrix = dok_matrix((len(order_ids), self.num_products), np.float32)
        interactions_matrix = dok_matrix((len(order_ids), self.num_products), np.float32)
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


class FitLightFMv2(_LightFM, FitModel):

    def run(self):
        self.random = RandomState(self.random_seed)

        orders_path = self.requires()['orders'].output().path
        _, user_features, interactions = self._generate_matrices(orders_path, self.max_prior_orders)

        model = LightFM(no_components=self.no_components,
                        loss=self.loss,
                        max_sampled=self.max_sampled,
                        random_state=self.random)
        model.fit(interactions, user_features=user_features, epochs=self.epochs,
                  num_threads=self.num_threads, verbose=True)

        joblib.dump(model, self.output().path)


class PredictLightFMv2ReorderSizeKnown(_LightFM, PredictModel):

    def requires(self):
        req = super().requires()
        req['model'] = FitLightFMv2(
            mode=self.mode,
            loss=self.loss,
            no_components=self.no_components,
            max_sampled=self.max_sampled)
        return req

    @staticmethod
    def _count_reordered_products(order):
        k = 0
        for product in order['products']:
            if product['reordered']:
                k += 1
        return k

    def _determine_reorder_size(self, orders_path):
        assert self.mode == 'evaluation'
        num_reordered = {}
        with open(orders_path) as orders_file:
            for line in orders_file:
                user_data = ujson.loads(line)
                order_id = int(user_data['last_order']['order_id'])
                num_reordered[order_id] = self._count_reordered_products(user_data['last_order'])
        return num_reordered

    def run(self):
        self.random = RandomState(self.random_seed)

        orders_path = self.requires()['orders'].output().path
        order_ids, user_features, _ = self._generate_matrices(orders_path, max_prior_orders=1)
        model = joblib.load(self.input()['model'].path)
        reorder_size = self._determine_reorder_size(orders_path)

        # Compute the score for each previously ordered product
        predictions = {}
        for i in range(len(order_ids)):
            order_id = order_ids[i]
            _, previously_ordered = user_features[i, :].nonzero()
            scores = model.predict(i, previously_ordered, user_features=user_features, num_threads=self.num_threads)
            df = pd.DataFrame({'product_id': previously_ordered + 1, 'score': scores})
            df = df.nlargest(reorder_size[order_id], 'score')
            predictions[order_id] = []
            for row in df.itertuples(index=False):
                # ujson fails when it tries to serialize the numpy int values
                predictions[order_id].append(int(row.product_id))

        with self.output().open('w') as fd:
            ujson.dump(predictions, fd)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
