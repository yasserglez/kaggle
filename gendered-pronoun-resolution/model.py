import math
from collections import OrderedDict
from typing import Optional, List, Tuple, Dict

import torch
from torch import nn
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, gelu
from allennlp.modules import ScalarMix
import numpy as np
import Levenshtein

from config import Config
from data import DATA_DIR, GAPExample, GAPLabel


def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight.data, nonlinearity='leaky_relu')
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
    elif isinstance(module, ScalarMix):
        for p in module.scalar_parameters:
            torch.nn.init.normal_(p.data)
        torch.nn.init.constant_(module.gamma, 1.0)


class BertFeatures(object):
    """
    Use BERT to extract features from a sequence of tokens.

    Returns:
      - A dictionary with the offsets of the span corresponding to
        each original token in the sequence of wordpiece tokens.
      - The outputs of all the hidden layers.
    """

    def __init__(self, model_name, cache_dir, device):
        self.do_lower_case = model_name.endswith('-uncased')
        self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()
        self.device = device
        self.model.to(self.device)

    def _wordpiece_tokenizer(self, tokens: List[str]) -> Tuple[List[str], Dict]:
        wordpiece_tokens: List[str] = []
        tokens_offsets: Dict[int, Tuple[int, int]] = OrderedDict()
        for token_index, token in enumerate(tokens):
            if self.do_lower_case:
                token = token.lower()
            split_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
            tokens_offsets[token_index] = \
                len(wordpiece_tokens), len(wordpiece_tokens) + len(split_tokens) - 1
            wordpiece_tokens.extend(split_tokens)
        # Add an entry for the end of the spans (which are not inclusive).
        tokens_offsets[len(tokens_offsets)] = len(wordpiece_tokens), len(wordpiece_tokens)
        return wordpiece_tokens, tokens_offsets

    def _prepare_input(self, wordpiece_tokens: List[str],
                       padding_length: int) -> Tuple[List[int], List[int], List[int]]:
        # The convention in BERT is:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        if len(wordpiece_tokens) > padding_length - 2:
            # Account for [CLS] and [SEP]
            wordpiece_tokens = wordpiece_tokens[0:(padding_length - 2)]
        wordpiece_tokens.insert(0, '[CLS]')
        wordpiece_tokens.append('[SEP]')
        input_type_ids = [0] * len(wordpiece_tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(wordpiece_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens.
        # Only real tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the input length.
        while len(input_ids) < padding_length:
            input_ids.append(0)
            input_type_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == padding_length
        assert len(input_type_ids) == padding_length
        assert len(input_mask) == padding_length
        return input_ids, input_type_ids, input_mask

    def generate(self, tokens_batch: List[List[str]]) -> Tuple[List[Dict], List[torch.Tensor]]:
        # Convert the input tokens to wordpiece tokens keeping track
        # of the spans corresponding to each original token.
        padding_length = 0
        wordpiece_tokens_batch, tokens_offsets_batch = [], []
        for tokens in tokens_batch:
            wordpiece_tokens, tokens_offsets = self._wordpiece_tokenizer(tokens)
            padding_length = max(padding_length, len(wordpiece_tokens))
            wordpiece_tokens_batch.append(wordpiece_tokens)
            tokens_offsets_batch.append(tokens_offsets)
        padding_length += 2  # Account for [CLS] and [SEP]

        input_ids_batch, input_type_ids_batch, input_mask_batch = [], [], []
        for wordpiece_tokens in wordpiece_tokens_batch:
            input_ids, input_type_ids, input_mask = \
                self._prepare_input(wordpiece_tokens, padding_length)
            input_ids_batch.append(input_ids)
            input_type_ids_batch.append(input_type_ids)
            input_mask_batch.append(input_mask)
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long).to(self.device)
        input_type_ids_batch = torch.tensor(input_type_ids_batch, dtype=torch.long).to(self.device)
        input_mask_batch = torch.tensor(input_mask_batch, dtype=torch.float).to(self.device)

        with torch.no_grad():
            hidden_outputs_batch, _ = self.model(
                input_ids_batch, input_type_ids_batch, input_mask_batch,
                output_all_encoded_layers=True)
        # Remove [CLS] and [SEP]
        hidden_outputs_batch = [h[:, 1:-1] for h in hidden_outputs_batch]
        return tokens_offsets_batch, hidden_outputs_batch


class FeedForward(nn.Module):
    """
    Feed-forward neural network.
    """

    def __init__(self, input_size: int, output_size: int,
                 hidden_layers: int = 0,
                 hidden_size: Optional[int] = None,
                 dropout: float = 0.0,
                 activation: str = 'relu'):
        super().__init__()

        if not hidden_size:
            hidden_size = (input_size + output_size) // 2
        units = [input_size] + [hidden_size] * hidden_layers + [output_size]
        self.layer_norm = nn.ModuleList()
        self.layer = nn.ModuleList()
        for x, y in zip(units, units[1:]):
            self.layer_norm.append(nn.LayerNorm(x))
            self.layer.append(nn.Linear(x, y))

        self.dropout = nn.Dropout(dropout)

        self.activation = {
            'relu': torch.nn.functional.relu,
            'leaky_relu': torch.nn.functional.leaky_relu,
            'elu': torch.nn.functional.elu,
            'gelu': gelu,
        }[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_norm, layer in zip(self.layer_norm, self.layer):
            x = self.activation(layer(self.dropout(layer_norm(x))))
        return x


class Model(nn.Module):

    def __init__(self, config: Config, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.device = device

        self.bert_features = BertFeatures(
            self.config.bert_model, DATA_DIR / 'bert', self.device)
        bert_config = self.bert_features.model.config

        self.mix = ScalarMix(mixture_size=bert_config.num_hidden_layers)

        self.coref = FeedForward(
            3 * bert_config.hidden_size + self.config.dist_size,
            self.config.coref_size,
            hidden_layers=(self.config.coref_layers - 1),
            dropout=self.config.dropout,
            activation='leaky_relu')

        # These values were selected based on the distances calculated
        # from the training data (5% fall on each bucket).
        self.dist_buckets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12,
                             13, 14, 16, 18, 20, 23, 26, 30, 37]
        self.dist = nn.Embedding(len(self.dist_buckets) + 1, self.config.dist_size)

        self.hidden = FeedForward(
            2 * self.config.coref_size + 6,
            self.config.hidden_size,
            hidden_layers=(self.config.hidden_layers - 1),
            dropout=self.config.dropout,
            activation='leaky_relu')

        self.classifier = nn.Linear(self.config.hidden_size, len(GAPLabel))

        self.apply(init_weights)

    def forward(self, batch: List[GAPExample]) -> torch.Tensor:
        # Use BERT as a features extractor. Calculate a linear combination
        # of the outputs of all the hidden layers as the input tokens.
        tokens_offsets, encoded_layers = self.bert_features.generate([e.tokens for e in batch])
        input_length = [next(reversed(offset.values()))[1] for offset in tokens_offsets]
        input_tokens = self.mix(encoded_layers)

        # Determine pronoun and mention positions in the BERT token sequence.
        pronoun_index = [tokens_offsets[i][e.pronoun_index][0] for i, e in enumerate(batch)]
        a_start = [tokens_offsets[i][e.a_start][0] for i, e in enumerate(batch)]
        a_end = [tokens_offsets[i][e.a_end][1] for i, e in enumerate(batch)]
        b_start = [tokens_offsets[i][e.b_start][0] for i, e in enumerate(batch)]
        b_end = [tokens_offsets[i][e.b_end][1] for i, e in enumerate(batch)]

        pronoun = input_tokens[range(input_tokens.size(0)), pronoun_index]
        context = self._calculate_context_vector(
            input_tokens, input_length, pronoun_index)

        a_span = list(zip(a_start, a_end))
        a = self._calculate_mention_vector(
            input_tokens, input_length, pronoun_index, a_span)
        a_distance = self._calculate_pronoun_mention_distance(
            [e.pronoun_index for e in batch],
            [(e.a_start, e.a_end) for e in batch])
        a_coref = self.coref(torch.cat([pronoun, a, context - a, a_distance], dim=-1))

        b_span = list(zip(b_start, b_end))
        b = self._calculate_mention_vector(
            input_tokens, input_length, pronoun_index, b_span)
        b_distance = self._calculate_pronoun_mention_distance(
            [e.pronoun_index for e in batch],
            [(e.b_start, e.b_end) for e in batch])
        b_coref = self.coref(torch.cat([pronoun, b, context - b, b_distance], dim=-1))

        features = self._calculate_features(batch)

        hidden_input = torch.cat([a_coref, b_coref, features], dim=-1)
        hidden_output = self.hidden(hidden_input)
        logits = self.classifier(hidden_output)
        return logits

    def _calculate_mention_vector(
            self, input_tokens: torch.Tensor,
            input_length: List[int],
            pronoun_index: List[int],
            spans: List[Tuple[int, int]]) -> torch.Tensor:
        # Mark positions that should be attended in each sequence with 1's
        mask = torch.zeros(input_tokens.size(0), input_tokens.size(1))
        span_length = torch.zeros(input_tokens.size(0))
        for i in range(input_tokens.size(0)):
            start, end = spans[i]
            mask[i, start:end] = 1.0
            span_length[i] = end - start
        masked_tokens = input_tokens * mask.unsqueeze(-1).to(self.device)

        # Average pooling over the selected positions. This generates
        # a fixed-length vector from each input sequence.
        result = masked_tokens.sum(dim=-2) / span_length.unsqueeze(-1).to(self.device)
        return result

    def _calculate_context_vector(
            self, input_tokens: torch.Tensor,
            input_length: List[int],
            pronoun_index: List[int]) -> torch.Tensor:
        pronoun = input_tokens[range(input_tokens.size(0)), pronoun_index]
        attention_scores = torch.matmul(
            pronoun.unsqueeze(-2),
            input_tokens.transpose(-1, -2)).squeeze(-2)
        dk = self.bert_features.model.config.hidden_size
        attention_scores /= math.sqrt(dk)

        # Attend to all valid positions except for the pronoun.
        attention_mask = torch.ones_like(attention_scores)
        for i in range(input_tokens.size(0)):
            attention_mask[i, input_length[i]:] = 0
            attention_mask[i, pronoun_index[i]] = 0

        # From pytorch_pretrained_bert: Since attention_mask is 1.0
        # for positions we want to attend and 0.0 for masked positions,
        # this operation will create a tensor which is 0.0 for positions
        # we want to attend and -10000.0 for masked positions. Since
        # we are adding it to the raw scores before the softmax, this
        # is effectively the same as removing these entirely.
        attention_mask = (1.0 - attention_mask) * -10000.0

        # Normalize the attention scores to probabilities.
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attended_tokens = input_tokens * attention_probs.unsqueeze(-1)
        context = attended_tokens.sum(dim=-2)
        return context

    def _calculate_pronoun_mention_distance(
            self, pronoun_index: List[int],
            mention_span: List[Tuple[int, int]]) -> torch.Tensor:
        distances = []
        for p, (start, end) in zip(pronoun_index, mention_span):
            m = start if p < start else (end - 1)
            distances.append(abs(p - m))
        indices = np.searchsorted(self.dist_buckets, distances)
        return self.dist(torch.LongTensor(indices).to(self.device))

    def _calculate_features(self, batch: List[GAPExample]) -> torch.Tensor:
        all_features = []
        for example in batch:
            mention_spans = [(example.a_start, example.a_end),
                             (example.b_start, example.b_end)]
            # Edit distances between the Wikipedia page name and each mention.
            edit_distances = []
            page_name = example.url.split('/')[-1].replace('_', ' ').lower()
            for start, end in mention_spans:
                mention = ' '.join(example.tokens[start:end]).lower()
                d1 = Levenshtein.ratio(mention, page_name)
                d2 = Levenshtein.jaro_winkler(mention, page_name, 0.25)
                d3 = Levenshtein.jaro_winkler(mention[::-1], page_name[::-1], 0.25)
                edit_distances.extend([d1, d2, d3])
            all_features.append(edit_distances)

        all_features = torch.FloatTensor(all_features).to(self.device)
        return all_features

    def calculate_loss(
            self, batch: List[GAPExample],
            reduction: str = 'mean') -> torch.Tensor:
        logits = self.forward(batch)
        targets = torch.LongTensor([e.label for e in batch]).to(self.device)
        loss_func = nn.CrossEntropyLoss(reduction=reduction)
        loss = loss_func(logits, targets)
        return loss
