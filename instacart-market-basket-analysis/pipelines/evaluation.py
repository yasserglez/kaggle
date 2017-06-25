import os

import luigi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import MODELS_DIR
from .clean_data import TestOrders
from .models import ModelPredictions


class EvaluateModel(luigi.Task):

    model_name = luigi.Parameter()

    def requires(self):
        return {
            'orders': TestOrders(),
            'predictions': ModelPredictions(mode='evaluation', model_name=self.model_name),
        }

    def output(self):
        csv_path = os.path.join(MODELS_DIR, 'evaluation', '{}.csv'.format(self.model_name))
        pdf_path = os.path.join(MODELS_DIR, 'evaluation', '{}.png'.format(self.model_name))
        return {'data': luigi.LocalTarget(csv_path), 'plot': luigi.LocalTarget(pdf_path)}

    def run(self):
        orders = self.requires()['orders'].read()
        predictions = self.requires()['predictions'].read()

        order_ids = []
        precision_values = []
        recall_values = []
        f1_values = []

        for user_data in orders:
            target_order = user_data['last_order']
            order_ids.append(target_order['order_id'])

            # Correct target values
            y_true = set()
            for product in target_order['products']:
                if product['reordered']:
                    y_true.add(product['product_id'])
            if not y_true:
                y_true.add('None')

            # Predictions
            y_pred = set(predictions[str(target_order['order_id'])])
            if not y_pred:
                y_pred.add('None')

            # Metrics
            precision = len(y_true & y_pred) / len(y_pred)
            recall = len(y_true & y_pred) / len(y_true)
            f1 = 2.0 * (precision * recall) / (precision + recall) if precision or recall else 0.0
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)

        data = pd.DataFrame({
            'model': [self.model_name for k in range(len(order_ids))],
            'order_id': order_ids,
            'precision': precision_values,
            'recall': recall_values,
            'f1': f1_values})
        data = data[['model', 'order_id', 'precision', 'recall', 'f1']]
        data.to_csv(self.output()['data'].path, index=False)

        new_columns = {}
        for metric in ['precision', 'recall', 'f1']:
            new_columns[metric] = '{}\nmean: {}'.format(metric, np.round(data[metric].mean(), 7))
        df = pd.melt(data.rename(columns=new_columns), value_vars=sorted(new_columns.values()), var_name='metric')
        plt.figure()
        ax = sns.boxplot(x='metric', y='value', data=df)
        ax.set_title(self.model_name)
        ax.set_xlabel('')
        ax.get_figure().savefig(self.output()['plot'].path)


class EvaluateAllModels(luigi.WrapperTask):

    def requires(self):
        for filename in os.listdir(os.path.join(MODELS_DIR, 'evaluation')):
            if filename.endswith('.json'):
                yield EvaluateModel(model_name=filename[:-5])


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
