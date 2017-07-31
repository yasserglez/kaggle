import pprint
import subprocess
import tempfile
from collections import defaultdict
from contextlib import contextmanager

import ujson
import numpy as np
from sklearn.utils import shuffle


def hidden_layer_units(num_layers, from_dim, to_dim):
    units = np.linspace(from_dim, to_dim, num_layers + 2)[1:-1]
    units = np.round(units, 0).astype(np.int)
    return units


class ExamplesGenerator(object):

    @staticmethod
    def _count_lines(file_path):
        p = subprocess.Popen(['wc', '-l', file_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return int(p.communicate()[0].partition(b' ')[0])

    @contextmanager
    def _open_shuffled(self, file_path):
        with tempfile.NamedTemporaryFile(delete=True) as f:
            subprocess.call(['shuf', file_path, '-o', f.name])
            yield open(f.name)

    def _generate_examples(self, last_order, prior_orders):
        raise NotImplementedError()

    def _generate_user_examples(self, user_data, max_prior_orders):
        yield from self._generate_examples(user_data['last_order'], user_data['prior_orders'])
        max_prior_orders -= 1
        if max_prior_orders > 0:
            for k in range(len(user_data['prior_orders']) - 1, 0, -1):
                last_order = user_data['prior_orders'][k]
                prior_orders = user_data['prior_orders'][:k]
                yield from self._generate_examples(last_order, prior_orders)
                max_prior_orders -= 1
                if max_prior_orders == 0:
                    break

    def _load_data(self, orders_path):
        order_ids = []
        inputs = defaultdict(list)
        predictions = []

        def add_example(order_id, product, orders, prediction):
            order_ids.append(order_id)
            inputs['product'].append(product)
            inputs['orders'].append(orders)
            predictions.append(prediction)

        with open(orders_path) as orders_file:
            for line in orders_file:
                user_data = ujson.loads(line)
                for order_id, product, orders, prediction in self._generate_user_examples(user_data, max_prior_orders=1):
                    add_example(order_id, product, orders, prediction)

        # Build the numpy arrays
        inputs['product'] = np.array(inputs['product'])
        inputs['orders'] = np.array(inputs['orders'])
        predictions = np.array(predictions)
        inputs['product'], inputs['orders'], predictions = \
            shuffle(inputs['product'], inputs['orders'], predictions, random_state=self.random)

        return order_ids, inputs, predictions

    def _create_data_generator(self, orders_path, max_prior_orders, batch_size):
        # Count the number of training examples
        num_examples = 0
        with open(orders_path) as orders_file:
            for line in orders_file:
                user_data = ujson.loads(line)
                for _ in self._generate_user_examples(user_data, max_prior_orders):
                    num_examples += 1

        batch_sizes = [len(a) for a in np.array_split(range(num_examples), num_examples / batch_size)]
        steps_per_epoch = len(batch_sizes)

        def generator():
            while True:
                current_step = 0
                product_inputs, orders_inputs, predictions = [], [], []
                with self._open_shuffled(orders_path) as orders_file:
                    for line in orders_file:
                        user_data = ujson.loads(line)
                        # Generate examples from this user's data
                        for order_id, product, orders, prediction in self._generate_user_examples(user_data, max_prior_orders):
                            product_inputs.append(product)
                            orders_inputs.append(orders)
                            predictions.append(prediction)
                        # Return inputs and predictions if we have enough examples
                        while len(predictions) >= 10 * batch_sizes[current_step]:
                            product_inputs, orders_inputs, predictions = \
                                shuffle(product_inputs, orders_inputs, predictions, random_state=self.random)
                            b = batch_sizes[current_step]
                            inputs = {'product': np.array(product_inputs[:b]), 'orders': np.array(orders_inputs[:b])}
                            yield inputs, np.array(predictions[:b])
                            del product_inputs[:b]
                            del orders_inputs[:b]
                            del predictions[:b]
                            current_step += 1
                # Flush the rest of the examples
                while current_step < steps_per_epoch:
                    b = batch_sizes[current_step]
                    inputs = {'product': np.array(product_inputs[:b]), 'orders': np.array(orders_inputs[:b])}
                    yield inputs, np.array(predictions[:b])
                    del product_inputs[:b]
                    del orders_inputs[:b]
                    del predictions[:b]
                    current_step += 1
                assert current_step == steps_per_epoch
                assert len(product_inputs) == 0
                assert len(orders_inputs) == 0
                assert len(predictions) == 0

        return generator(), steps_per_epoch
