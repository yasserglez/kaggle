import os
from collections import defaultdict

import luigi
import ujson
import numpy as np
from numpy.random import RandomState
import pandas as pd

from .config import OUTPUT_DIR
from .input_data import OrdersInput, OrderProductsInput


class _OrdersTask(luigi.Task):

    def requires(self):
        return {
            'orders': OrdersInput(),
            'order_products': [OrderProductsInput(eval_set=s) for s in ('prior', 'train')],
        }

    def _read_orders_input(self):
        dtype = {
            'order_id': np.uint32,
            'user_id': np.uint32,
            'eval_set': str,
            'order_number': np.uint8,
            'order_dow': np.uint8,
            'order_hour_of_day': np.uint8,
            'days_since_prior_order': np.float16,
        }
        df = pd.read_csv(self.input()['orders'].path, dtype=dtype)
        return df

    def _read_order_products_input(self):
        dtype = {
            'order_id': np.uint32,
            'product_id': np.uint32,
            'add_to_cart_order': np.uint8,
            'reordered': np.uint8,
        }
        df_parts = []
        for task in self.input()['order_products']:
            df_part = pd.read_csv(task.path, dtype=dtype)
            df_parts.append(df_part)
        df = pd.concat(df_parts)
        return df

    def _write_orders(self, orders):
        orders_by_user = defaultdict(list)

        current_order = None
        orders.sort_values(['user_id', 'order_number', 'add_to_cart_order'], inplace=True)
        for row in orders.itertuples(index=False):
            user_id = int(row.user_id)
            if current_order != row.order_id:
                current_order = row.order_id
                order = dict(
                    order_id=int(row.order_id),
                    day_of_week=int(row.order_dow),
                    hour_of_day=int(row.order_hour_of_day),
                    days_since_prior_order=None,
                    products=None)
                if not np.isnan(row.days_since_prior_order):
                    order['days_since_prior_order'] = int(row.days_since_prior_order)
                orders_by_user[user_id].append(order)
            if not np.isnan(row.product_id):
                product = dict(product_id=int(row.product_id), reordered=int(row.reordered))
                if not orders_by_user[user_id][-1]['products']:
                    orders_by_user[user_id][-1]['products'] = list()
                orders_by_user[user_id][-1]['products'].append(product)

        with self.output().open('w') as fd:
            for user_id, orders in orders_by_user.items():
                user_data = dict(user_id=user_id, last_order=orders[-1], prior_orders=orders[:-1])
                fd.write(ujson.dumps(user_data) + '\n')

    def read(self):
        orders = []
        with open(self.output().path) as fd:
            for line in fd:
                user_data = ujson.loads(line)
                orders.append(user_data)
        return orders


class CompleteOrders(_OrdersTask):

    def output(self):
        path = os.path.join(OUTPUT_DIR, 'complete_orders.json')
        return luigi.LocalTarget(path)

    def run(self):
        orders = self._read_orders_input()
        # Ignore the orders from the test set that don't have product information.
        orders = orders[orders.eval_set != 'test']
        order_products = self._read_order_products_input()
        clean_orders = orders.merge(order_products, how='inner', on='order_id')
        del orders, order_products
        self._write_orders(clean_orders)


class SubmissionOrders(CompleteOrders):

    def output(self):
        path = os.path.join(OUTPUT_DIR, 'submission_orders.json')
        return luigi.LocalTarget(path)

    def run(self):
        orders = self._read_orders_input()
        order_products = self._read_order_products_input()
        test_users = set(orders[orders.eval_set == 'test'].user_id.unique())
        test_orders = orders[orders.user_id.isin(test_users)]
        data = test_orders.merge(order_products, how='left', on='order_id')
        del orders, order_products, test_users, test_orders
        self._write_orders(data)


class _SplitOrdersTask(_OrdersTask):

    random_seed = luigi.IntParameter(default=758140847)
    test_size = luigi.FloatParameter(default=0.2)

    def requires(self):
        return CompleteOrders()

    def _run(self, op):
        rng = RandomState(self.random_seed)
        with self.output().open('w') as output_fd:
            with self.input().open('r') as input_fd:
                for line in input_fd:
                    if op(rng.uniform(), self.test_size):
                        output_fd.write(line)


class TestOrders(_SplitOrdersTask):

    def output(self):
        filename = 'test_orders_{}_{}.json'.format(self.test_size, self.random_seed)
        return luigi.LocalTarget(os.path.join(OUTPUT_DIR, filename))

    def run(self):
        super()._run(np.less_equal)


class TrainingOrders(_SplitOrdersTask):

    def output(self):
        filename = 'training_orders_{}_{}.json'.format(1.0 - self.test_size, self.random_seed)
        return luigi.LocalTarget(os.path.join(OUTPUT_DIR, filename))

    def run(self):
        super()._run(np.greater)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
