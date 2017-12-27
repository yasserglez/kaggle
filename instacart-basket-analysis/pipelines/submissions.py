import os

import luigi
import pandas as pd

from .config import SUBMISSIONS_DIR
from .models import ModelPredictions


class Submission(luigi.Task):

    model_name = luigi.Parameter()

    def requires(self):
        return {
            'predictions': ModelPredictions(mode='submission', model_name=self.model_name),
        }

    def output(self):
        submission = self.__class__.__name__.lower()
        path = os.path.join(SUBMISSIONS_DIR, '{}_{}.csv'.format(submission, self.model_name))
        return luigi.LocalTarget(path)

    def run(self):
        predictions = self.requires()['predictions'].read()

        order_ids, products = [], []
        for order_id, product_list in predictions.items():
            order_ids.append(order_id)
            products.append(' '.join(str(p) for p in product_list) if product_list else 'None')

        submission = pd.DataFrame({'order_id': order_ids, 'products': products}, columns=['order_id', 'products'])
        submission.to_csv(self.output().path, index=False)


class Submission1(Submission):
    model_name = 'empty'


class Submission2(Submission):
    model_name = 'frequent_products_0.9'


class Submission3(Submission):
    model_name = 'previous_order_reordered'


class Submission4(Submission):
    model_name = 'mlp_v1_3_relu_0.5_0.1'


class Submission5(Submission):
    model_name = 'mlp_v2_5_91_5_3_0.5_threshold_0.173'


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
