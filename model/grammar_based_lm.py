import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import multiprocessing

import pandas as pd

import typing

from embedding.wordembedding import load_vocabulary, Vocabulary
from read_data.load_parsed_data import get_token_vocabulary, get_vocabulary_id_map

BEGIN, END, UNK = ["<BEGIN>", "<END>", "<UNK>"]
PAD_TOKEN = -1


class CCodeDataLoader(Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 transform=None):
        self.data_df = data_df[data_df['parse_tree'].map(lambda x: x is not None)]
        self.transform = transform

    def __getitem__(self, index):
        sample = {"tree": self.data_df.iloc[index]["parse_tree"],
                  "tokens": self.data_df.iloc[index]["tokens"],
                  "length": len(self.data_df.iloc[index]["tokens"])}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data_df)


class TokenIdMap(object):
    """
    Map the Token to the corresponding id
    """
    def __init__(self, vocabulary: Vocabulary):
        self._vocabulary = vocabulary

    def __call__(self, sample):
        """
        :param sample: a list of tokens
        :return:
        """
        return [self._vocabulary.word_to_id(token) for token in sample]


class GrammarLanguageModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_state_size,
                 rnn_num_layers,
                 batch_size):
        super().__init__()
        self._batch_size = batch_size
        self._rnn_num_layers = rnn_num_layers
        self._hidden_state_size = hidden_state_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_TOKEN, sparse=True).cpu()
        self.rnn = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_state_size,
                           num_layers=rnn_num_layers,).cuda()
        self._initial_state = self.initial_state()

    def _embedding(self, token_sequence):
        """
        :param token_sequence: a long variable with the shape [batch, seq]
        :return: The embedding seq of shape [batch, seq, embedding_dim]
        """
        return self.embeddings(token_sequence).cuda()

    def initial_state(self):
        return (autograd.Variable(torch.randn((self._rnn_num_layers, self._batch_size, self._hidden_state_size))),
                autograd.Variable(torch.randn((self._rnn_num_layers, self._batch_size, self._hidden_state_size))))

    def _forward_rnn(self,
                     embedding_sequence,
                     lengths):
        """
        :param embedding_sequence: a float variable with the shape [batch, seq, embedding_dim]
        :param lengths: a long variable with the shape [batch, ]
        :return: a float variable with the shape [batch, seq, feature]
        """
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(embedding_sequence, lengths, batch_first=True)
        output, _ = self.rnn(packed_seq, self._initial_state)
        return output

    def forward(self,
                token_sequence,
                token_type_sequence,
                production_seq,
                target_seq,
                lengths):
        """
        :param token_sequence: a long variable with the shape [batch, seq]
        :param token_type_sequence: a long variable with the shape [batch, seq]
        :param production_seq: a long variable with the shape [batch, seq]
        :param target_seq: a long variable with the shape [batch, seq]
        :param lengths: a long variable with the shape [batch, ]
        :return: a float variable with the shape [batch, seq, vovabulary_size]
        """
        embedding_feature = self._embedding(token_sequence)
        rnn_feature = self._forward_rnn(embedding_feature, lengths)


if __name__ == '__main__':
    vocabulary = load_vocabulary(get_token_vocabulary, get_vocabulary_id_map, [BEGIN], [END], UNK)

