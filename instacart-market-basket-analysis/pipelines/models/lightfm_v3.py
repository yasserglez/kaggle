import pprint

import luigi
import ujson
import numpy as np
from numpy.random import RandomState
from scipy.sparse import dok_matrix
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from lightfm import LightFM
from lightfm.evaluation import auc_score

from ..models import FitModel
from ..clean_data import Products


class FitLightFMv3Embeddings(FitModel):

    # Use all the available data to train the embeddings
    mode = 'submission'

    items = luigi.ChoiceParameter(choices=['departments', 'aisles', 'products'], default='products')
    no_components = luigi.IntParameter(default=10)

    random_seed = luigi.IntParameter(default=3996193, significant=False)
    epochs = luigi.IntParameter(default=1000, significant=False)
    patience = luigi.IntParameter(default=10, significant=False)
    num_threads = luigi.IntParameter(default=16, significant=False)

    @property
    def model_name(self):
        params = [self.items, self.no_components]
        model_name = 'lightfm_v3_embeddings_{}'.format('_'.join(str(p) for p in params))
        return model_name

    def read(self):
        lightfm_model = joblib.load(self.output().path)
        biases, embeddings = lightfm_model.get_item_representations()
        return embeddings

    def _load_products(self):
        if not hasattr(self, '_num_products'):
            products = Products.read()
            self._num_departments = len(products.department_id.unique())
            self._num_aisles = len(products.aisle_id.unique())
            self._num_products = products.shape[0]
            self._product_id_to_aisle_id = dict(zip(products.product_id, products.aisle_id))
            self._product_id_to_department_id = dict(zip(products.product_id, products.department_id))

    def _get_num_items(self):
        self._load_products()
        if self.items == 'departments':
            return self._num_departments
        elif self.items == 'aisles':
            return self._num_aisles
        elif self.items == 'products':
            return self._num_products

    def _get_item_id(self, product_id):
        self._load_products()
        if self.items == 'departments':
            return self._product_id_to_department_id[product_id] - 1
        elif self.items == 'aisles':
            return self._product_id_to_aisle_id[product_id] - 1
        elif self.items == 'products':
            return product_id - 1

    def _generate_row(self, last_order, prior_orders):
        # Collect the indices of the previously ordered items
        previously_ordered = set()
        for order in prior_orders:
            for product in order['products']:
                item = self._get_item_id(product['product_id'])
                previously_ordered.add(item)
        # Collect the indices of the reordered items
        reordered = set()
        for product in last_order['products']:
            if product['reordered']:
                item = self._get_item_id(product['product_id'])
                reordered.add(item)
        return last_order['order_id'], previously_ordered, reordered

    def _generate_rows(self, user_data, num_orders):
        yield self._generate_row(user_data['last_order'], user_data['prior_orders'])
        num_orders -= 1
        if num_orders > 0:
            for k in range(len(user_data['prior_orders']) - 1, 0, -1):
                last_order = user_data['prior_orders'][k]
                prior_orders = user_data['prior_orders'][:k]
                yield self._generate_row(last_order, prior_orders)
                num_orders -= 1
                if num_orders == 0:
                    break

    def _generate_matrices(self, orders_path):
        order_ids, previously_ordered_sets, reordered_sets = [], [], []

        # Collect the data for the sparse matrices
        with open(orders_path) as f:
            for line in f:
                user_data = ujson.loads(line)
                for order_id, previously_ordered, reordered in self._generate_rows(user_data, 1):
                    order_ids.append(order_id)
                    previously_ordered_sets.append(previously_ordered)
                    reordered_sets.append(reordered)

        # Populate the sparse matrices
        num_items = self._get_num_items()
        user_features_matrix = dok_matrix((len(order_ids), num_items), np.float32)
        interactions_matrix = dok_matrix((len(order_ids), num_items), np.float32)
        for i in range(len(order_ids)):
            for j in previously_ordered_sets[i]:
                user_features_matrix[i, j] = 1
                if j in reordered_sets[i]:
                    # Previously ordered and reordered => positive interaction
                    interactions_matrix[i, j] = 1
                else:
                    # Previously ordered but did not reorder => negative interaction
                    interactions_matrix[i, j] = -1
        user_features_matrix = user_features_matrix.tocsr()
        interactions_matrix = interactions_matrix.tocoo()

        return order_ids, user_features_matrix, interactions_matrix

    def run(self):
        self.random = RandomState(self.random_seed)

        orders_path = self.requires()['orders'].output().path
        _, features, interactions = self._generate_matrices(orders_path)

        train_features, val_features, train_interactiosn, val_interactions = \
            train_test_split(features, interactions, test_size=0.1, random_state=self.random)

        model = LightFM(loss='logistic', no_components=self.no_components, random_state=self.random)

        wait = 0
        best_val_auc = None
        for epoch in range(1, self.epochs + 1):
            model.fit_partial(train_interactiosn, user_features=train_features,
                              epochs=self.epochs, num_threads=self.num_threads)
            auc_scores = auc_score(model, val_interactions, user_features=val_features,
                                   num_threads=self.num_threads)
            current_val_auc = np.nan_to_num(auc_scores).mean()
            if best_val_auc is None or current_val_auc > best_val_auc:
                joblib.dump(model, self.output().path)
                best_val_auc = current_val_auc
                wait = 0
            else:
                wait += 1
                if wait == self.patience:
                    break
            print('Epoch {}/{} - AUC: {:.6g}'.format(epoch, self.epochs, best_val_auc))


class FitAllLightFMv3Embeddings(luigi.Task):

    def run(self):
        yield FitLightFMv3Embeddings(items='departments', no_components=3)
        yield FitLightFMv3Embeddings(items='aisles', no_components=5)
        yield FitLightFMv3Embeddings(items='products', no_components=10)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
