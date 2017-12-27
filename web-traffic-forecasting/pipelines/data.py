import os
import csv
import multiprocessing
from multiprocessing.pool import ThreadPool

import luigi
import numpy as np
import pandas as pd
from numpy.random import RandomState

from .config import RANDOM_SEED, INPUT_DIR, OUTPUT_DIR


class Key(luigi.ExternalTask):
    stage = luigi.IntParameter(default=1)

    def output(self):
        path = os.path.join(INPUT_DIR, 'key_{}.csv'.format(self.stage))
        return luigi.LocalTarget(path)

    def read(self):
        df = pd.read_csv(self.output().path)
        return df


class InputData(luigi.ExternalTask):
    stage = luigi.IntParameter(default=1)

    def output(self):
        path = os.path.join(INPUT_DIR, 'train_{}.csv'.format(self.stage))
        return luigi.LocalTarget(path)

    def read(self):
        df = pd.read_csv(self.output().path)
        return df


class ImputedData(luigi.Task):
    stage = luigi.IntParameter(default=1)
    method = luigi.ChoiceParameter(default='zeros', choices=['zeros', 'linear', 'quadratic'])

    def requires(self):
        return InputData(stage=self.stage)

    def output(self):
        path = os.path.join(OUTPUT_DIR, 'train_{}_imputed_{}.csv'.format(self.stage, self.method))
        return luigi.LocalTarget(path)

    def read(self):
        df = pd.read_csv(self.output().path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    def run(self):
        data = self.requires().read()

        # Transform to long format
        data = pd.melt(data, id_vars='Page', var_name='Date', value_name='Views')
        data['Date'] = pd.to_datetime(data['Date'])

        if self.method == 'zeros':
            data.fillna(0, inplace=True)
        else:
            num_pages = len(data['Page'].unique())
            page_groups = (page_data for page, page_data in data.groupby('Page'))
            cpu_count = multiprocessing.cpu_count()
            with ThreadPool(cpu_count) as pool:
                parts = list(pool.imap_unordered(
                    self._interpolate, page_groups,
                    chunksize=num_pages // (4 * cpu_count)))
            data = pd.concat(parts)

        data = data[['Page', 'Date', 'Views']]
        data.to_csv(self.output().path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    def _interpolate(self, page_data):
        page_data = page_data.set_index('Date')
        missing = np.isnan(page_data['Views'].values)
        k = 0

        # Replace missing values at the beginning of the time series with zeros
        if missing[k]:
            while k < len(missing) and missing[k]:
                k += 1
            page_data.iloc[:k, page_data.columns.get_loc('Views')] = 0

        # Use linear interpolation to fill the other missing values
        if any(missing[k:]):
            page_data = page_data.interpolate(method=self.method)
            page_data['Views'] = np.maximum(0, np.round(page_data['Views']))

        return page_data.reset_index()


class ImputedDataSample(luigi.Task):
    stage = luigi.IntParameter(default=1)
    method = luigi.ChoiceParameter(default='zeros', choices=['zeros', 'linear', 'quadratic'])
    sample_ratio = luigi.FloatParameter(default=0.1)
    random_seed = luigi.IntParameter(default=RANDOM_SEED)

    def requires(self):
        return ImputedData(stage=self.stage, method=self.method)

    def output(self):
        filename = 'train_{}_imputed_{}_sample_{}_{}.csv' \
            .format(self.stage, self.method, self.sample_ratio, self.random_seed)
        path = os.path.join(OUTPUT_DIR, filename)
        return luigi.LocalTarget(path)

    def read(self):
        df = pd.read_csv(self.output().path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    def run(self):
        data = self.requires().read()
        pages = list(sorted(data['Page'].unique()))

        rng = RandomState(self.random_seed)
        sample_size = int(np.round(len(pages) * self.sample_ratio))
        pages_sample = set(rng.choice(pages, sample_size, replace=False))
        data_sample = data[data['Page'].isin(pages_sample)]

        data_sample.to_csv(self.output().path, index=False)


def get_data_dir(stage, imputation, sample_ratio, random_seed, deploy_date, from_date, to_date):
    parts = [
        'train_{}'.format(stage),
        imputation,
        sample_ratio,
        random_seed,
        deploy_date,
        from_date,
        to_date,
    ]
    return '_'.join(str(p) for p in parts)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
