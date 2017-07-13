from collections import defaultdict, Counter

import luigi
import ujson
import numpy as np
import pandas as pd
from numpy.random import RandomState

from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from ..models import FitModel, PredictModel
from ..clean_data import Aisles, Products


class _MLPv1(object):

    random_seed = luigi.IntParameter(default=3996193)

    num_hidden_layers = luigi.IntParameter(default=3)
    activation = luigi.Parameter(default='relu')
    dropout = luigi.FloatParameter(default=0.5)

    @property
    def model_name(self):
        params = [self.num_hidden_layers, self.activation, self.dropout]
        model_name = 'mlp_v1_{}'.format('_'.join(str(p) for p in params))
        if getattr(self, 'threshold', None) is not None:
            model_name += '_{}'.format(self.threshold)
        return model_name

    def _generate_user_features(self, orders):
        values = defaultdict(dict)

        # Customer value: https://en.wikipedia.org/wiki/RFM_(customer_value)
        r, f, m = {}, {}, {}
        for user_data in orders:
            user_id = user_data['user_id']
            days_since_prior_order = user_data['last_order']['days_since_prior_order']
            # Recency: use days_since_prior_order from the order to be predicted
            assert days_since_prior_order <= 30
            r[user_id] = (30 - days_since_prior_order) / 30
            # Frequency: total number of orders
            f[user_id] = len(user_data['prior_orders'])
            # Monetary value: average basket size
            basket_sizes = []
            for order in user_data['prior_orders']:
                basket_sizes.append(len(order['products']))
            m[user_id] = np.mean(basket_sizes)

        max_f = max(f.values())
        max_m = max(m.values())

        for user_data in orders:
            user_id = user_data['user_id']
            # Customer value
            values[user_id]['customer_value'] = np.array([r[user_id], f[user_id] / max_f, m[user_id]] / max_m)
            # Day of the week
            values[user_id]['day_of_week'] = to_categorical(user_data['last_order']['day_of_week'], 7)[0]
            # Time of the day: https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
            hour_of_day = user_data['last_order']['hour_of_day']
            sin_time = np.sin(2 * np.pi * hour_of_day / 23)
            cos_time = np.cos(2 * np.pi * hour_of_day / 23)
            values[user_id]['time_of_day'] = np.array([sin_time, cos_time])

        return values

    def _generate_product_features(self, orders):
        values = defaultdict(dict)

        products = Products.read()

        num_aisles = Aisles.count()
        for row in products.itertuples(index=False):
            values[row.product_id]['aisle'] = to_categorical(row.aisle_id - 1, num_aisles)[0]

        order_count = Counter()
        product_frequencies = defaultdict(list)
        for user_data in orders:
            num_days = 0
            product_timestamp = defaultdict(list)
            for order in user_data['prior_orders']:
                days_since_prior_order = order['days_since_prior_order']
                if days_since_prior_order is None:
                    days_since_prior_order = 0
                num_days += days_since_prior_order
                for product in order['products']:
                    order_count[product['product_id']] += 1
                    product_timestamp[product['product_id']].append(num_days)
            for product_id, timestamps in product_timestamp.items():
                if len(timestamps) > 1:
                    mean_freq = np.diff(timestamps).mean()
                    product_frequencies[product_id].append(mean_freq)

        max_order_count = max(order_count)

        for product_id in products.product_id:
            # Product popularity: number of orders that include the product
            values[product_id]['product_popularity'] = order_count[product_id] / max_order_count
            # Product frequency: how often is the product reordered?
            median_freq = np.median(product_frequencies.get(product_id, [0]))
            values[product_id]['product_frequency'] = median_freq / 365

        return values

    def _generate_user_product_features(self, orders):
        values = defaultdict(lambda: defaultdict(dict))

        # Product preference: how many times has the user ordered the product?
        for user_data in orders:
            user_id = user_data['user_id']
            order_count = Counter()
            for order in user_data['prior_orders']:
                for product in order['products']:
                    order_count[product['product_id']] += 1
            # Normalization
            for product_id in order_count:
                values[user_id][product_id]['user_product_preference'] = \
                    order_count[product_id] / len(user_data['prior_orders'])

        # Product recency: number of days since the user last ordered the product.
        for user_data in orders:
            user_id = user_data['user_id']
            num_days = 0
            for order in reversed(user_data['prior_orders']):
                days_since_prior_order = order['days_since_prior_order']
                if days_since_prior_order is None:
                    days_since_prior_order = 0
                num_days += days_since_prior_order
                for product in order['products']:
                    if product['product_id'] not in values[user_id]:
                        values[user_id][product['product_id']] = num_days / 365

        # Product frequency: how often does the user reorder the product?
        for user_data in orders:
            user_id = user_data['user_id']
            num_days = 0
            order_days = defaultdict(list)
            for order in user_data['prior_orders']:
                days_since_prior_order = order['days_since_prior_order']
                if days_since_prior_order is None:
                    days_since_prior_order = 0
                num_days += days_since_prior_order
                for product in order['products']:
                    order_days[product['product_id']].append(num_days)
            for product_id, timestamps in order_days.items():
                if len(timestamps) > 1:
                    user_value = np.diff(timestamps).mean() / 365
                    values[user_id][product_id]['user_product_frequency'] = user_value

        return values

    def _load_data(self):
        order_ids = []
        product_ids = []
        inputs = []
        predictions = []

        orders = self.requires()['orders'].read()
        user_features = self._generate_user_features(orders)
        product_features = self._generate_product_features(orders)
        user_product_features = self._generate_user_product_features(orders)

        def add_example(user_id, order_id, product_id, prediction):
            order_ids.append(order_id)
            product_ids.append(product_id)
            feature_vector = []
            # User features
            feature_vector.extend(user_features[user_id]['customer_value'])
            feature_vector.extend(user_features[user_id]['day_of_week'])
            feature_vector.extend(user_features[user_id]['time_of_day'])
            # Product features
            feature_vector.extend(product_features[product_id]['aisle'])
            feature_vector.append(product_features[product_id]['product_popularity'])
            feature_vector.append(product_features[product_id]['product_frequency'])
            # User and product features
            feature_vector.append(user_product_features[user_id][product_id].get('user_product_preference', 0))
            feature_vector.append(user_product_features[user_id][product_id].get('user_product_recency', 0))
            feature_vector.append(user_product_features[user_id][product_id].get('user_product_frequency', 0))
            inputs.append(np.array(feature_vector))
            predictions.append(prediction)

        for user_data in orders:
            user_id = user_data['user_id']
            order_id = user_data['last_order']['order_id']
            added_products = set()
            if user_data['last_order']['products']:
                for product in user_data['last_order']['products']:
                    # Ignore products that the user ordered for the first time
                    if product['reordered']:
                        add_example(user_id, order_id, product['product_id'], prediction=1)
                        added_products.add(product['product_id'])
            for order in user_data['prior_orders']:
                for product in order['products']:
                    if product['product_id'] not in added_products:
                        add_example(user_id, order_id, product['product_id'], prediction=0)
                        added_products.add(product['product_id'])

        return order_ids, product_ids, inputs, predictions


class FitMLPv1(_MLPv1, FitModel):

    def run(self):
        self.random = RandomState(self.random_seed)

        order_ids, product_ids, inputs, predictions = self._load_data()
        training_data, validation_data = self._split_data(order_ids, product_ids, inputs, predictions, training_size=0.8)
        _, training_inputs, training_predictions = training_data
        _, validation_inputs, validation_predictions = validation_data
        del order_ids, product_ids, inputs, predictions

        model = self._build_model(training_inputs)
        model.summary()

        model.fit(training_inputs, training_predictions,
            validation_data=(validation_inputs, validation_predictions),
            batch_size=1024, epochs=1000, verbose=2,
            callbacks=[EarlyStopping(min_delta=1e-5, patience=10)])

        model.save(self.output().path)

    def _split_data(self, order_ids, product_ids, inputs, predictions, training_size):
        training_order_ids = []
        training_product_ids = []
        training_inputs = []
        training_predictions = []

        validation_order_ids = []
        validation_product_ids = []
        validation_inputs = []
        validation_predictions = []

        current_order = None
        for i in range(len(order_ids)):
            if current_order != order_ids[i]:
                order_selected = self.random.uniform() <= training_size
                current_order = order_ids[i]
            if order_selected:
                training_order_ids.append(order_ids[i])
                training_product_ids.append(product_ids[i])
                training_inputs.append(inputs[i])
                training_predictions.append(predictions[i])
            else:
                validation_order_ids.append(order_ids[i])
                validation_product_ids.append(product_ids[i])
                validation_inputs.append(inputs[i])
                validation_predictions.append(predictions[i])

        training_inputs = np.array(training_inputs)
        validation_inputs = np.array(validation_inputs)
        training_data = training_order_ids, training_inputs, np.array(training_predictions)
        validation_data = validation_order_ids, validation_inputs, np.array(validation_predictions)

        return training_data, validation_data

    @staticmethod
    def _tapered_layers(from_dim, to_dim, num_layers):
        values = np.linspace(from_dim, to_dim, num=num_layers + 2)[1:-1]
        return [int(v) for v in values]

    def _build_model(self, training_inputs):
        inputs_dim = training_inputs.shape[1]
        all_inputs = Input(shape=(inputs_dim, ), name='all_inputs')

        hidden_layer = all_inputs
        for units in self._tapered_layers(inputs_dim, 1, self.num_hidden_layers):
            hidden_layer = Dropout(self.dropout)(hidden_layer)
            hidden_layer = Dense(units, activation=self.activation)(hidden_layer)

        hidden_layer = Dropout(self.dropout)(hidden_layer)
        prediction = Dense(1, activation='sigmoid', name='prediction')(hidden_layer)

        model = Model(inputs=all_inputs, outputs=prediction)
        model.compile(optimizer='adam', loss='binary_crossentropy')

        return model


class PredictMLPv1(_MLPv1, PredictModel):

    threshold = luigi.FloatParameter(default=0.1)

    def requires(self):
        req = super().requires()
        req['model'] = FitMLPv1(
            mode=self.mode,
            num_hidden_layers=self.num_hidden_layers,
            activation=self.activation,
            dropout=self.dropout)
        return req

    def run(self):
        order_ids, product_ids, inputs, _ = self._load_data()
        inputs = np.array(inputs)

        model = load_model(self.input()['model'].path)
        model.summary()

        scores = model.predict(inputs, batch_size=1024).flatten()
        df = pd.DataFrame({'order_id': order_ids, 'product_id': product_ids, 'score': scores})
        df = df[df.score > self.threshold].sort_values('score', ascending=False)

        predictions = {}
        for order_id in order_ids:
            predictions[order_id] = []
        for row in df.itertuples(index=False):
            # ujson fails when it tries to serialize the numpy int values
            predictions[int(row.order_id)].append(int(row.product_id))

        with self.output().open('w') as fd:
            ujson.dump(predictions, fd)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
