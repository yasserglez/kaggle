import os

import luigi
import ujson

from ..config import MODELS_DIR
from ..clean_data import CompleteOrders, SubmissionOrders, TrainingOrders, TestOrders


class _ModelTask(luigi.Task):

    mode = luigi.ChoiceParameter(choices=['evaluation', 'submission'], default='evaluation')


class FitModel(_ModelTask):

    def requires(self):
        return {
            'orders': TrainingOrders() if self.mode == 'evaluation' else CompleteOrders(),
        }

    @property
    def model_name(self):
        raise NotImplementedError

    def output(self):
        path = os.path.join(MODELS_DIR, self.mode, self.model_name)
        return luigi.LocalTarget(path)


class PredictModel(_ModelTask):

    def requires(self):
        return {
            'orders': TestOrders() if self.mode == 'evaluation' else SubmissionOrders(),
        }

    @property
    def model_name(self):
        raise NotImplementedError

    def output(self):
        basename = '{}_{}'.format(self.model_name, self.threshold) if self.threshold else self.model_name
        path = os.path.join(MODELS_DIR, self.mode, '{}.json'.format(basename))
        return luigi.LocalTarget(path)


class ModelPredictions(luigi.ExternalTask):

    model_name = luigi.Parameter()
    mode = luigi.ChoiceParameter(choices=['evaluation', 'submission'], default='evaluation')

    def output(self):
        path = os.path.join(MODELS_DIR, self.mode, '{}.json'.format(self.model_name))
        return luigi.LocalTarget(path)

    def read(self):
        with self.output().open() as fd:
            return ujson.load(fd)
