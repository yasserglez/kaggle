import os

import luigi
import ujson

from ..config import MODELS_DIR
from ..clean_data import CompleteOrders, SubmissionOrders, TrainingOrders, TestOrders


class FitModel(luigi.Task):

    training_mode = luigi.ChoiceParameter(choices=['evaluation', 'submission'], default='submission')

    def requires(self):
        return {
            'orders': TrainingOrders() if self.training_mode == 'evaluation' else CompleteOrders(),
        }

    @property
    def model_name(self):
        raise NotImplementedError

    def output(self):
        path = os.path.join(MODELS_DIR, self.training_mode, self.model_name)
        return luigi.LocalTarget(path)


class PredictModel(luigi.Task):

    prediction_mode = luigi.ChoiceParameter(choices=['evaluation', 'submission'], default='submission')

    def requires(self):
        return {
            'orders': TestOrders() if self.prediction_mode == 'evaluation' else SubmissionOrders(),
        }

    @property
    def model_name(self):
        raise NotImplementedError

    def output(self):
        path = os.path.join(MODELS_DIR, self.prediction_mode, '{}.json'.format(self.model_name))
        return luigi.LocalTarget(path)


class ModelPredictions(luigi.ExternalTask):

    training_mode = luigi.ChoiceParameter(choices=['evaluation', 'submission'], default='submission')
    model_name = luigi.Parameter()

    def output(self):
        path = os.path.join(MODELS_DIR, self.training_mode, '{}.json'.format(self.model_name))
        return luigi.LocalTarget(path)

    def read(self):
        with self.output().open() as fd:
            return ujson.load(fd)
