import os

import luigi

from .config import INPUT_DIR


class OrdersInput(luigi.ExternalTask):

    def output(self):
        path = os.path.join(INPUT_DIR, 'orders.csv')
        return luigi.LocalTarget(path)


class OrderProductsInput(luigi.ExternalTask):

    eval_set = luigi.Parameter()

    def output(self):
        path = os.path.join(INPUT_DIR, 'order_products__{}.csv'.format(self.eval_set))
        return luigi.LocalTarget(path)

