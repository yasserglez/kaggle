import luigi
import ujson

from ..models import PredictModel


# Predict an empty order for each user.


class PredictEmpty(PredictModel):

    @property
    def model_name(self):
        return 'empty'

    def run(self):
        orders = self.requires()['orders'].read()
        predictions = {}
        for user_data in orders:
            predictions[user_data['last_order']['order_id']] = []
        with self.output().open('w') as fd:
            ujson.dump(predictions, fd)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
