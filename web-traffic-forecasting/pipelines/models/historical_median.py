import csv

import luigi
import numpy as np
import pandas as pd

from ..models import FitModel, PredictModel


class FitHistoricalMedian(FitModel):

    model_name = 'historical_median'

    def run(self):
        data = self.requires()['data'].read()
        data = data[data['Date'] <= self.deploy_date]

        pages, medians = [], []
        grouped_data = {page: page_data for page, page_data in data.groupby('Page')}
        for page, page_data in grouped_data.items():
            pages.append(page)
            page_data = page_data.sort_values('Date')
            try:
                first_nonzero = page_data['Views'].nonzero()[0][0]
                median = page_data.iloc[first_nonzero:, :]['Views'].median()
            except IndexError:
                median = 0.0
            medians.append(median)
        df = pd.DataFrame({'Page': pages, 'Median': medians}, columns=['Page', 'Median'])
        df.to_csv(self.output().path, index=False, quoting=csv.QUOTE_NONNUMERIC)


class PredictHistoricalMedian(PredictModel):

    model_name = 'historical_median'

    def requires(self):
        req = super().requires()
        req['model'] = FitHistoricalMedian(
            stage=self.stage,
            imputation=self.imputation,
            sample_ratio=self.sample_ratio,
            deploy_date=self.deploy_date,
            from_date=self.from_date,
            to_date=self.to_date,
            random_seed=self.random_seed)
        return req

    def run(self):
        data = self.requires()['data'].read()

        pages = data['Page'].unique()
        dates = pd.date_range(start=self.from_date, end=self.to_date)
        medians = pd.read_csv(self.input()['model'].path)
        predictions = pd.DataFrame({
            'Page': np.repeat(pages, len(dates)),
            'Date': np.tile(dates, len(pages)),
        })
        predictions = predictions.merge(medians, how='left', on='Page').fillna(0)
        predictions.rename(columns={'Median': 'Prediction'}, inplace=True)
        predictions = predictions[['Page', 'Date', 'Prediction']]
        predictions.to_csv(self.output().path, index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
