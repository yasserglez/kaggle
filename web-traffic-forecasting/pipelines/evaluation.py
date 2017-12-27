import os

import luigi
import pandas as pd
import numpy as np
from dateutil.parser import parse

from .config import OUTPUT_DIR
from .data import InputData
from .models import ModelPredictions


# https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/38481
def calculate_smape(y_true, y_pred):
    valid_datapoints = ~np.isnan(y_true)
    y_true, y_pred = y_true[valid_datapoints], y_pred[valid_datapoints]
    raw_smape = np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
    kaggle_smape = np.nan_to_num(raw_smape)
    return np.mean(kaggle_smape) * 200


class ModelEvaluation(luigi.Task):
    data_dir = luigi.Parameter()
    model_name = luigi.Parameter()

    def requires(self):
        return {
            'data': InputData(stage=3),
            'predictions': ModelPredictions(data_dir=self.data_dir, model_name=self.model_name),
        }

    def output(self):
        path = os.path.join(OUTPUT_DIR, 'evaluation', self.data_dir, '{}.csv'.format(self.model_name))
        dirname = os.path.dirname(path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        return luigi.LocalTarget(path)

    def run(self):
        from_date, to_date = self.data_dir.split('_')[-2:]
        from_date = parse(from_date)
        to_date = parse(to_date)

        predictions = self.requires()['predictions'].read()
        pages = set(predictions['Page'].unique())
        predictions = predictions.sort_values(['Page', 'Date'])
        y_pred = predictions['Prediction'].values

        data = self.requires()['data'].read()
        data = data[data['Page'].isin(pages)]
        data = pd.melt(data, id_vars='Page', var_name='Date', value_name='Views')
        data['Date'] = pd.to_datetime(data['Date'])
        data = data[(data['Date'] >= from_date) & (data['Date'] <= to_date)]
        data = data.sort_values(['Page', 'Date'])
        y_true = data['Views'].values

        smape = calculate_smape(y_true, y_pred)

        df = pd.DataFrame([[self.data_dir, self.model_name, smape]], columns=['Data', 'Model', 'sMAPE'])
        df.to_csv(self.output().path, index=False)


class EvaluationSummary(luigi.Task):

    def complete(self):
        return False

    def output(self):
        path = os.path.join(OUTPUT_DIR, 'evaluation')
        return luigi.LocalTarget(os.path.join(path, 'summary.csv'))

    def requires(self):
        model_evaluations = []
        path = os.path.join(OUTPUT_DIR, 'predictions')
        for data_dir in os.listdir(path):
            if not data_dir.startswith('train_'):
                continue
            for model_file in os.listdir(os.path.join(path, data_dir)):
                if not model_file.endswith('.csv'):
                    continue
                model_name = model_file[:-4]
                results = ModelEvaluation(data_dir=data_dir, model_name=model_name)
                model_evaluations.append(results)
        return model_evaluations

    def run(self):
        df_parts = []
        for results in self.requires():
            df_part = pd.read_csv(results.output().path)
            df_parts.append(df_part)
        df = pd.concat(df_parts)
        df.to_csv(self.output().path, index=False)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
