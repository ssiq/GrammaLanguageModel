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

from c_code_processer.code_util import parse_tree_to_top_down_process, ProductionVocabulary, \
    get_all_c99_production_vocabulary, LeafToken
from common.util import generate_mask, show_process_map
from embedding.wordembedding import load_vocabulary, Vocabulary
from read_data.load_parsed_data import get_token_vocabulary, get_vocabulary_id_map, read_parsed_tree_code

BEGIN, END, UNK = ["<BEGIN>", "<END>", "<UNK>"]
PAD_TOKEN = -1


class CCodeDataSet(Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary: Vocabulary,
                 transform=None):
        self.data_df = data_df[data_df['parse_tree'].map(lambda x: x is not None)]
        self.transform = transform
        self.vocabulary = vocabulary

        self._samples = [self._get_raw_sample(i) for i in range(len(self.data_df))]
        if self.transform:
            self._samples = show_process_map(self.transform, self._samples)

    def _get_raw_sample(self, index):
        tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["tokens"]]],
                                                        use_position_label=True)[0]
        sample = {"tree": self.data_df.iloc[index]["parse_tree"],
                  "tokens": tokens[:-1],
                  "target": tokens[1:],
                  "length": len(tokens)-1}
        return sample

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)


class GrammarLanguageModelTypeInputMap(object):
    """
    Map the top down parsing order node to the input format of the GrammarLanguageModel
    """
    def __init__(self,
                 production_vocabulary: ProductionVocabulary):
        self._production_vocabulary = production_vocabulary

    def __call__(self, sample):
        """
        :param sample: a list of node
        :return: a dict of list {'to_parse_token', 'terminal_mask'}
        """
        production_vocabulary = self._production_vocabulary

        get_token_id = production_vocabulary.get_token_id
        get_matched_terminal_index = production_vocabulary.get_matched_terminal_node
        vocabulary_size = production_vocabulary.token_num()
        generate_mask_fn = generate_mask(size=vocabulary_size)
        get_node_right_id = lambda x: x.right_id

        stack = [sample[0].left_id]
        to_parse_token_id = [sample[0].left_id]

        for node in sample:
            type_id = stack.pop()
            assert type_id == node.left_id
            if isinstance(node, LeafToken):
                to_parse_token_id.append(stack[-1])
            else:
                for right_id in reversed(get_node_right_id(node)):
                    stack.append(right_id)

        terminal_mask = [generate_mask_fn(get_matched_terminal_index(token)) for token in to_parse_token_id]
        to_parse_token_id.append(get_token_id(production_vocabulary.EMPTY), )
        terminal_mask.append(generate_mask_fn(get_token_id(production_vocabulary.EMPTY)))
        return {"to_parse_token": to_parse_token_id, "terminal_mask": terminal_mask}


class FlatMap(object):
    """
    This map the sample dict to a flat map
    """
    def __call__(self, sample: dict):
        res = {}

        def add_(d: dict):
            for k, v in d.items():
                if not isinstance(v, dict):
                    res[k] = v
                else:
                    add_(v)
        add_(sample)
        return res


class GrammarLanguageModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 type_num,
                 embedding_dim,
                 hidden_state_size,
                 rnn_num_layers,
                 batch_size):
        super().__init__()
        self._batch_size = batch_size
        self._rnn_num_layers = rnn_num_layers
        self._hidden_state_size = hidden_state_size

        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_TOKEN, sparse=True).cpu()
        self.type_embedding = nn.Embedding(type_num, embedding_dim, padding_idx=PAD_TOKEN, sparse=True).cpu()
        self.rnn = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_state_size,
                           num_layers=rnn_num_layers,).cuda()
        self._initial_state = self.initial_state()

    def _embedding(self, token_sequence):
        """
        :param token_sequence: a long variable with the shape [batch, seq]
        :return: The embedding seq of shape [batch, seq, embedding_dim]
        """
        return self.token_embeddings(token_sequence).cuda()

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


def key_transform(transform, key, ):
    def transform_fn(sample):
        sample[key] = transform(sample[key])
        return sample

    return transform_fn


def train_and_evaluate(data):
    for d, n in zip(data, ["train", "val", "test"]):
        print("There are {} raw data in the {} dataset".format(len(d), n))
    vocabulary = load_vocabulary(get_token_vocabulary, get_vocabulary_id_map, [BEGIN], [END], UNK)
    production_vocabulary = get_all_c99_production_vocabulary()
    transforms_fn = transforms.Compose([
        key_transform(GrammarLanguageModelTypeInputMap(production_vocabulary), "tree"),
        FlatMap(),
    ])
    generate_dataset = lambda df: CCodeDataSet(df, vocabulary, transforms_fn)
    data = [generate_dataset(d) for d in data]
    for d, n in zip(data, ["train", "val", "test"]):
        print("There are {} parsed data in the {} dataset".format(len(d), n))

