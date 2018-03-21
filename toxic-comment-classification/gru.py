import re
import logging
from collections import Counter
import multiprocessing as mp

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import Iterator
from unidecode import unidecode

import spacy
english = spacy.load('en')

import common
import base


logger = logging.getLogger(__name__)


def minimal_preprocessing(row):
    text = row[1]
    # Convert to ASCII
    text = unidecode(text)
    # Lowercase
    text = text.lower()
    # Replace sequences of whitespace characters with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    # Collapse all characters repeated 4+ times to 3 occurrences
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)

    words = []
    for token in english.tokenizer(text):
        word = token.text
        # Ignore spaces
        if not word.isspace():
            words.append(word)
    if not words:
        words.append('<UNK>')

    return row[0], words


class GRUModule(base.BaseModule):

    def __init__(self, vocab, annotation_dropout, prediction_dropout):
        super().__init__(vocab)

        embedding_size = vocab.vectors.shape[1]

        self.rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=embedding_size,
            bidirectional=True,
            batch_first=True)

        for name, param in self.rnn.named_parameters():
            if name.startswith('weight_ih_'):
                nn.init.xavier_uniform(param)
            elif name.startswith('weight_hh_'):
                nn.init.orthogonal(param)
            elif name.startswith('bias_'):
                nn.init.constant(param, 0.0)

        self.annotation = base.Dense(
            3 * embedding_size, embedding_size,
            hidden_layers=1,
            hidden_nonlinearity='relu',
            output_nonlinearity='tanh',
            dropout=annotation_dropout)

        self.label_vectors = nn.Parameter(torch.zeros(len(common.LABELS), embedding_size))
        nn.init.uniform(self.label_vectors, -1, 1)

        self.prediction = base.Dense(
            embedding_size, 1,
            output_nonlinearity='sigmoid',
            dropout=prediction_dropout)

    def forward(self, text, text_lengths):
        vectors = self.embedding(text)

        packed_vectors = pack_padded_sequence(vectors, text_lengths.tolist(), batch_first=True)
        packed_rnn_output, _ = self.rnn(packed_vectors)
        rnn_output, _ = pad_packed_sequence(packed_rnn_output, batch_first=True)

        annotations = torch.cat([vectors, rnn_output], -1)
        annotations = self.annotation(annotations)

        predictions = []
        for i in range(len(common.LABELS)):
            # Similarity between the word vectors and the context vector
            alpha = torch.matmul(annotations, self.label_vectors[i])
            # Reshape back into the sequence shape and normalize the coefficient
            alpha = alpha.view(annotations.shape[0], annotations.shape[1])
            alpha = F.softmax(alpha, dim=-1)
            # Scale each vector by its coefficient and sum over the sequence
            comment_vector = torch.sum(alpha.unsqueeze(-1) * annotations, dim=1)
            prediction = self.prediction(comment_vector)
            predictions.append(prediction)

        predictions = torch.cat(predictions, dim=-1)
        return predictions


class GRU(base.BaseModel):

    def load_preprocessed_data(self):
        preprocessed_data = {}

        logger.info('Preprocessing the comments')
        raw_data = common.load_raw_data()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.imap_unordered(minimal_preprocessing, raw_data.items(), chunksize=100)
            for k, (id_, words) in enumerate(results, start=1):
                preprocessed_data[id_] = words
                if k % 10000 == 0 or k == len(raw_data):
                    logger.info('Preprocessed {:,} / {:,} comments'.format(k, len(raw_data)))

        logger.info('Selecting the vocabulary')
        occurrences = Counter()
        lengths = []
        for id_, words in preprocessed_data.items():
            occurrences.update(set(words))
            lengths.append(len(words))
        logger.info('The original vocabulary size is {:,}'.format(len(occurrences)))
        logger.info('Number of words -- median: %d, 75%%: %d, 90%%: %d, 95%%: %d, 99%%: %d, max: %d',
                    *np.percentile(lengths, [50, 75, 90, 95, 99, 100]))
        vocab = set(word for word, count in occurrences.most_common(self.params['vocab_size'] - 1))

        logger.info('Replacing out-of-vocabulary words with <UNK>')
        for id_, words in preprocessed_data.items():
            new_words = [w if w in vocab else '<UNK>' for w in words]
            preprocessed_data[id_] = ' '.join(new_words[:self.params['max_len']])

        return preprocessed_data

    def build_train_iterator(self, df):
        dataset = base.CommentsDataset(df, self.fields)
        train_iter = Iterator(
            dataset, batch_size=self.params['batch_size'],
            repeat=False, sort_within_batch=True, shuffle=True,
            sort_key=lambda x: len(x.text),
            device=None if torch.cuda.is_available() else -1)
        return train_iter

    def build_prediction_iterator(self, df):
        dataset = base.CommentsDataset(df, self.fields)
        # Reorder the examples (required by pack_padded_sequence)
        sort_indices = sorted(range(len(dataset)), key=lambda i: -len(dataset[i].text))
        pred_id = [df['id'].iloc[i] for i in sort_indices]
        dataset.examples = [dataset.examples[i] for i in sort_indices]
        pred_iter = Iterator(
            dataset, batch_size=self.params['batch_size'],
            repeat=False, shuffle=False, sort=False,
            device=None if torch.cuda.is_available() else -1)
        return pred_id, pred_iter

    def build_model(self):
        model = GRUModule(
            vocab=self.vocab,
            annotation_dropout=self.params['annotation_dropout'],
            prediction_dropout=self.params['prediction_dropout'])
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def update_parameters(self, model, optimizer, loss):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        clip_grad_norm(parameters, 0.2)
        optimizer.step()


if __name__ == '__main__':
    params = {
        'vocab_size': 100000,
        'max_len': 300,
        'vectors': 'glove.twitter.27B.200d',
        'annotation_dropout': 0.1,
        'prediction_dropout': 0.3,
        'batch_size': 256,
        'lr_high': 0.5,
        'lr_low': 0.1,
    }
    model = GRU(params, random_seed=base.RANDOM_SEED)
    model.main()
