import pprint
from collections import Counter

import luigi
import ujson
import numpy as np

from ..models import PredictModel


# This generates a list of products for each user, ranked by how often they are reordered.
# It also counts the number of reordered products in each order, so we can get an estimate
# of how many products each user usually reorders. The model then recommends a group of
# products that are frequently reordered by the user.


class PredictFrequentProducts(PredictModel):

    reorder_percentile = luigi.FloatParameter(default=0.9)

    @property
    def model_name(self):
        return 'frequent_products_{}'.format(self.reorder_percentile)

    def run(self):
        orders = self.requires()['orders'].read()
        predictions = {}

        for user_data in orders:
            product_counts = Counter()
            reorder_sizes = []
            for order_num, order in enumerate(user_data['prior_orders']):
                reorder_size = 0
                for product_order in order['products']:
                    if product_order['reordered']:
                        product_counts[product_order['product_id']] += 1
                        reorder_size += 1
                if order_num > 0:
                    # Don't count the first order, which will always be zero.
                    reorder_sizes.append(reorder_size)
            all_products = [product_id for product_id, count in product_counts.most_common()]
            num_selected_products = int(np.percentile(reorder_sizes, 100 * self.reorder_percentile))
            predictions[user_data['last_order']['order_id']] = all_products[:num_selected_products]

        with self.output().open('w') as fd:
            ujson.dump(predictions, fd)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
