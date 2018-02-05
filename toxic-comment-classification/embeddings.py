import os
import sys
import pprint
import logging
import tempfile
import multiprocessing as mp
from collections import Counter

import joblib
import fastText
import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchtext.vocab import Vectors


import common


logger = logging.getLogger(__name__)


BASE_DIR = os.path.join(common.DATA_DIR, 'embeddings')
if not os.path.isdir(BASE_DIR):
    os.makedirs(BASE_DIR)


def load(preprocessed_data, params):
    """Load the pretrained embeddings.

    Parameters:
        embedding_size: size of word vectors.
        pretrain_model: 'skipgram' or 'cbow'.
        pretrain_lr: learning rate.
        pretrain_epochs: number of epochs.
    """
    param_names = ('embedding_size', 'pretrain_model', 'pretrain_lr', 'pretrain_epochs')
    params = {k: params[k] for k in param_names}
    if params['pretrain_model'] in {'skipgram', 'cbow'}:
        return load_fasttext(preprocessed_data, params)
    elif params['pretrain_model'] == 'pmi':
        return load_pmi(preprocessed_data, params)
    elif params['pretrain_model'] == 'lm':
        return LMVectors()


def load_fasttext(preprocessed_data, params):
    output_file = os.path.join(BASE_DIR, common.params_str(params))

    if os.path.isfile(output_file):
        logger.info(f'Loading {output_file[len(common.DATA_DIR) + 1:]}')
        model = fastText.load_model(output_file)
        return FastTextVectors(model)

    logger.info(f'Generating {output_file[len(common.DATA_DIR) + 1:]}')
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        for comment_text in preprocessed_data.values():
            tmp.write((comment_text + '\n').encode('utf-8'))
        tmp.flush()
        model = fastText.train_unsupervised(
            tmp.name,
            model=params['pretrain_model'],
            lr=params['pretrain_lr'],
            dim=params['embedding_size'],
            epoch=params['pretrain_epochs'],
            minCount=1,
            thread=mp.cpu_count(),
            verbose=0)
    model.save_model(output_file)
    return FastTextVectors(model)


class FastTextVectors(Vectors):
    """
    Wrapper for fastText's model to match the torchtext API.
    """

    def __init__(self, model):
        self.model = model
        self.itos = list(enumerate(model.get_words()))
        self.stoi = {word: i for i, word in enumerate(self.itos)}
        self.dim = model.get_dimension()

    def __getitem__(self, word):
        vector = self.model.get_word_vector(word)
        return torch.from_numpy(vector)


def load_pmi(preprocessed_data, params):
    output_file = os.path.join(BASE_DIR, common.params_str(params))

    if os.path.isfile(output_file):
        logger.info(f'Loading {output_file[len(common.DATA_DIR) + 1:]}')
        vectors = joblib.load(output_file)
        return vectors

    logger.info(f'Generating {output_file[len(common.DATA_DIR) + 1:]}')

    # Load the training data and keep only examples with positive labels
    examples = common.load_data('submission', None, 'train.csv')
    examples = examples[examples[common.LABELS].sum(axis=1) > 0]
    examples['text'] = examples['id'].map(preprocessed_data)

    logger.info('Building the PMI matrix')
    matrix, stoi = build_pmi_matrix(examples)

    # Iterate over all the word occurrences with positive labels
    # and collect the corresponding target value from the PMI matrix.
    word_indices = []
    label_indices = []
    target_values = []
    for text, labels in zip(examples['text'], examples[common.LABELS].values):
        j_values = list(np.flatnonzero(labels))
        for word in text.split():
            i = stoi[word]
            for j in j_values:
                if matrix[i, j] > 0:
                    word_indices.append(i)
                    label_indices.append(j)
                    target_values.append(matrix[i, j])

    logger.info('Generated {:,} training examples'.format(len(word_indices)))

    word_indices = torch.LongTensor(np.array(word_indices, dtype=np.int))
    label_indices = torch.LongTensor(np.array(label_indices, dtype=np.int))
    data_tensor = torch.stack([word_indices, label_indices], dim=1)
    target_tensor = torch.DoubleTensor(np.array(target_values))
    dataset = TensorDataset(data_tensor, target_tensor)
    loader = DataLoader(dataset, batch_size=16384, shuffle=True)

    model = PMIModel(len(stoi), params['embedding_size'])
    model = model.double().cuda()

    optimizer = optim.Adam(model.parameters(), lr=params['pretrain_lr'])

    for epoch in range(1, params['pretrain_epochs'] + 1):
        loss_sum = 0
        model.train()
        for data, target in loader:
            optimizer.zero_grad()
            data = autograd.Variable(data).cuda()
            target = autograd.Variable(target).cuda()
            output = model(data, target)
            loss = F.mse_loss(output, target)
            loss_sum += loss.data[0]
            loss.backward()
            optimizer.step()
        logger.info('Epoch {:04d}/{:04d} - loss: {:.6g}'
                    .format(epoch, params['pretrain_epochs'],
                            loss_sum / len(loader)))

    vectors = PMIVectors(stoi, model.word_embedding.weight.data.cpu().numpy())
    joblib.dump(vectors, output_file)
    return vectors


def build_pmi_matrix(examples):
    # Build the word vocabulary and count co-occurrences (i: word, j: label)
    stoi = {}
    ij_freq = Counter()
    for text, labels in zip(examples['text'], examples[common.LABELS].values):
        j_values = list(np.flatnonzero(labels))
        for word in text.split():
            if word not in stoi:
                stoi[word] = len(stoi)
            i = stoi[word]
            for j in j_values:
                ij_freq[(i, j)] += 1

    # Marginals (i: token, j: label)
    i_freq, j_freq = Counter(), Counter()
    for (i, j), count in ij_freq.items():
        i_freq[i] += count
        j_freq[j] += count

    D = sum(ij_freq.values())
    data, row_ind, col_ind = [], [], []
    for i, j in ij_freq.keys():
        value = np.log(D * ij_freq[(i, j)] / (i_freq[i] * j_freq[j]))
        row_ind.append(i)
        col_ind.append(j)
        data.append(value)

    matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(stoi), len(common.LABELS)))
    matrix.data[matrix.data < 0] = 0
    matrix.eliminate_zeros()

    return matrix, stoi


class PMIModel(nn.Module):

    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        nn.init.uniform(self.word_embedding.weight, -0.05, 0.05)
        self.label_embedding = nn.Embedding(len(common.LABELS), embedding_size)
        nn.init.uniform(self.label_embedding.weight, -0.05, 0.05)

    def forward(self, data, target):
        word_vectors = self.word_embedding(data[:, 0])
        label_vectors = self.label_embedding(data[:, 1])
        output = torch.bmm(word_vectors.unsqueeze(-2), label_vectors.unsqueeze(-1)).squeeze()
        return output


class PMIVectors(Vectors):
    """
    Word vectors pretrained based on predicting the cooccurrence of each word
    with the labels via the positive pointwise mutual information (PMI) matrix.
    """

    def __init__(self, stoi, vectors):
        self.stoi = stoi
        self.itos = [None] * vectors.shape[0]
        for word, i in stoi.items():
            self.itos[i] = word
        self.vectors = vectors
        self.dim = self.vectors.shape[1]

    def __getitem__(self, word):
        if word in self.stoi:
            vector = self.vectors[self.stoi[word]]
        else:
            vector = np.zeros_like(self.vectors[0])
        return torch.from_numpy(vector)


class LMVectors(Vectors):

    def __init__(self):
        checkout_dir = os.path.abspath(os.path.join(common.DATA_DIR, '../awd-lstm-lm'))
        sys.path.insert(0, checkout_dir)

        dictionary = joblib.load(os.path.join(checkout_dir, 'dictionary.pickle'))
        self.itos = dictionary.idx2word
        self.stoi = dictionary.word2idx
        model = torch.load(os.path.join(checkout_dir, 'model.pickle'))
        self.vectors = model.encoder.weight.data.cpu().numpy()
        self.dim = self.vectors.shape[1]

        sys.path.pop(0)

    def __getitem__(self, word):
        if word in self.stoi:
            vector = self.vectors[self.stoi[word]]
        else:
            vector = np.zeros_like(self.vectors[0])
        return torch.from_numpy(vector)
