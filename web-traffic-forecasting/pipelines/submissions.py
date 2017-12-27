import os
from datetime import date

import luigi
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, RANDOM_SEED
from .data import Key, get_data_dir
from .models import ModelPredictions


class Submission(luigi.Task):
    stage = luigi.IntParameter()
    predictions = luigi.ListParameter()

    def requires(self):
        predictions = []
        for path in self.predictions:
            data_dir, model_name = path.split('/')
            model_name = model_name[:-4]
            predictions.append(ModelPredictions(data_dir=data_dir, model_name=model_name))
        return {'key': Key(stage=self.stage), 'predictions': predictions}

    def output(self):
        submission_name = self.__class__.__name__.lower()
        path = os.path.join(OUTPUT_DIR, 'submissions', '{}.csv.gz'.format(submission_name))
        dirname = os.path.dirname(path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        return luigi.LocalTarget(path)

    def run(self):
        predictions = []
        for p in self.requires()['predictions']:
            predictions.append(p.read())

        if len(predictions) > 1:
            # Use the median of median of the individual predictions
            predictions = pd.concat(predictions)
            predictions = predictions.groupby(['Page', 'Date'], as_index=False).median()
        else:
            predictions = predictions[0]

        key = self.requires()['key'].read()
        predictions['Page'] = predictions['Page'] + '_' + predictions['Date'].dt.strftime('%Y-%m-%d')
        submission = predictions.merge(key, on='Page', how='left')
        submission['Visits'] = np.maximum(np.round(submission['Prediction']), 0).astype(int)

        submission = submission[['Id', 'Visits']].sort_values('Id')
        submission.to_csv(self.output().path, index=False, compression='gzip')


class Submission1(Submission):
    stage = 1
    predictions = [
        'train_1_zeros_1.0_7329199_2016-12-31_2017-01-01_2017-03-01/historical_median.csv',
    ]


class Submission2(Submission):
    stage = 2
    predictions = [
        'train_2_linear_1.0_2766306141_2017-08-31_2017-09-13_2017-11-13/rnn_v1_30_3_2_selu_0.5_smape_0.001_3.csv',
        'train_2_linear_1.0_2517504021_2017-08-31_2017-09-13_2017-11-13/rnn_v1_90_3_2_selu_0.5_smape_0.001_3.csv',
        'train_2_linear_1.0_1532850297_2017-08-31_2017-09-13_2017-11-13/rnn_v1_180_3_2_selu_0.5_smape_0.001_3.csv',
        'train_2_linear_1.0_1406174211_2017-08-31_2017-09-13_2017-11-13/rnn_v1_270_3_2_selu_0.5_smape_0.001_3.csv',
        'train_2_linear_1.0_4221485063_2017-08-31_2017-09-13_2017-11-13/rnn_v1_365_3_2_selu_0.5_smape_0.001_3.csv',
    ]


class Submission3(Submission):
    stage = 2
    predictions = [
        'train_2_linear_1.0_1659827432_2017-08-31_2017-09-13_2017-11-13/rnn_v1_30_3_2_selu_0.5_smape_0.001_3.csv',
        'train_2_linear_1.0_308308759_2017-08-31_2017-09-13_2017-11-13/rnn_v1_30_3_2_selu_0.5_smape_0.001_3.csv',
        'train_2_linear_1.0_3927851431_2017-08-31_2017-09-13_2017-11-13/rnn_v1_30_3_2_selu_0.5_smape_0.001_3.csv',
        'train_2_linear_1.0_4230874159_2017-08-31_2017-09-13_2017-11-13/rnn_v1_30_3_2_selu_0.5_smape_0.001_3.csv',
        'train_2_linear_1.0_911303249_2017-08-31_2017-09-13_2017-11-13/rnn_v1_30_3_2_selu_0.5_smape_0.001_3.csv',
    ]


class Submission4(Submission):
    stage = 3
    predictions = [
        'train_3_linear_1.0_1226457454_2017-09-10_2017-09-13_2017-11-13/rnn_v1_30_3_2_selu_0.5_smape_0.001_3.csv',
        'train_3_linear_1.0_196017792_2017-09-10_2017-09-13_2017-11-13/rnn_v1_30_3_2_selu_0.5_smape_0.001_3.csv',
        'train_3_linear_1.0_2693560142_2017-09-10_2017-09-13_2017-11-13/rnn_v1_30_3_2_selu_0.5_smape_0.001_3.csv',
        'train_3_linear_1.0_3391346194_2017-09-10_2017-09-13_2017-11-13/rnn_v1_30_3_2_selu_0.5_smape_0.001_3.csv',
        'train_3_linear_1.0_4208396372_2017-09-10_2017-09-13_2017-11-13/rnn_v1_30_3_2_selu_0.5_smape_0.001_3.csv',
        'train_3_linear_1.0_535029318_2017-09-10_2017-09-13_2017-11-13/rnn_v1_30_3_2_selu_0.5_smape_0.001_3.csv',
        'train_3_linear_1.0_96302290_2017-09-10_2017-09-13_2017-11-13/rnn_v1_30_3_2_selu_0.5_smape_0.001_3.csv',
    ]


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
