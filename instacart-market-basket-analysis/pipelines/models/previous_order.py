import luigi
import ujson

from ..models import PredictModel


class PredictPreviousOrder(PredictModel):

    products = luigi.ChoiceParameter(choices=['all', 'reordered'], default='all')

    @property
    def model_name(self):
        return 'previous_order_{}'.format(self.products)

    def run(self):
        orders = self.requires()['orders'].read()
        predictions = {}
        for user_data in orders:
            user_products = []
            for product in user_data['prior_orders'][-1]['products']:
                if self.products == 'all':
                    user_products.append(product['product_id'])
                elif self.products == 'reordered' and product['reordered']:
                    user_products.append(product['product_id'])
            order_id = user_data['last_order']['order_id']
            predictions[order_id] = user_products

        with self.output().open('w') as fd:
            ujson.dump(predictions, fd)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
