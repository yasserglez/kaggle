import os
import pprint

import luigi
import numpy as np
from sklearn.externals import joblib
import spacy

from ..models import FitModel
from ..clean_data import Products


class ProductNameVectors(FitModel):

    mode = 'submission'

    embedding_dim = luigi.IntParameter(default=50)

    @property
    def model_name(self):
        model_name = 'product_name_vectors_{}'.format(self.embedding_dim)
        return model_name

    def read(self):
        product_name_vectors = joblib.load(self.output().path)
        return product_name_vectors

    def run(self):
        nlp = spacy.load('en')
        glove_dir = os.path.join(os.path.dirname(__file__), 'glove')
        glove_file = os.path.join(glove_dir, 'glove.6B.{}d.txt'.format(self.embedding_dim))
        nlp.vocab.load_vectors(open(glove_file))

        def get_vector(product_name):
            vector = np.zeros(self.embedding_dim)
            num_words = 0
            doc = nlp(product_name.lower())
            for word in doc:
                if not (word.is_oov or word.is_stop or word.is_digit or word.is_punct):
                    vector += word.vector
                    num_words += 1
            if num_words > 0:
                vector /= num_words
            return vector

        products = Products.read()
        product_name_vectors = np.array(products['product_name'].map(get_vector).tolist())
        joblib.dump(product_name_vectors, self.output().path)


class FitAllProductNameVectors(luigi.Task):

    def run(self):
        for n in [50, 100, 200, 300]:
            yield ProductNameVectors(embedding_dim=n)


if __name__ == '__main__':
    luigi.run(local_scheduler=True)
