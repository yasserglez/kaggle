import os
from datetime import date

import luigi
import pandas as pd

from ..config import OUTPUT_DIR, RANDOM_SEED
from ..data import ImputedData, ImputedDataSample, get_data_dir


class _ModelTask(luigi.Task):
    stage = luigi.IntParameter(default=2)
    imputation = luigi.Parameter(default='linear')
    sample_ratio = luigi.FloatParameter(default=1.0)
    deploy_date = luigi.DateParameter(default=date(2017, 8, 31))
    from_date = luigi.DateParameter(default=date(2017, 9, 13))
    to_date = luigi.DateParameter(default=date(2017, 11, 13))
    random_seed = luigi.IntParameter(default=RANDOM_SEED)

    def requires(self):
        assert 0 <= self.sample_ratio <= 1
        if self.sample_ratio < 1:
            data = ImputedDataSample(
                stage=self.stage, method=self.imputation,
                sample_ratio=self.sample_ratio, random_seed=self.random_seed)
        else:
            data = ImputedData(stage=self.stage, method=self.imputation)
        return {'data': data}

    @property
    def data_dir(self):
        return get_data_dir(
            self.stage,
            self.imputation,
            self.sample_ratio,
            self.random_seed,
            self.deploy_date,
            self.from_date,
            self.to_date)


class FitModel(_ModelTask):

    @property
    def model_name(self):
        raise NotImplementedError

    def output(self):
        path = os.path.join(OUTPUT_DIR, 'models', self.data_dir, self.model_name)
        dirname = os.path.dirname(path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        return luigi.LocalTarget(path)


class PredictModel(_ModelTask):

    @property
    def model_name(self):
        raise NotImplementedError

    def output(self):
        path = os.path.join(OUTPUT_DIR, 'predictions', self.data_dir, '{}.csv'.format(self.model_name))
        dirname = os.path.dirname(path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        return luigi.LocalTarget(path)


class ModelPredictions(luigi.ExternalTask):
    data_dir = luigi.Parameter()
    model_name = luigi.Parameter()

    def output(self):
        path = os.path.join(OUTPUT_DIR, 'predictions', self.data_dir, '{}.csv'.format(self.model_name))
        return luigi.LocalTarget(path)

    def read(self):
        df = pd.read_csv(self.output().path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
