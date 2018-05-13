import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import pandas as pd

import os

import config
from c_code_processer.code_util import LeafToken
from c_code_processer.slk_parser import SLKProductionVocabulary, C99LabelVocabulary
from common.constants import pre_defined_c_tokens_map
from common.util import show_process_map, generate_mask, padded_to_length
from embedding.wordembedding import Vocabulary

BEGIN, END, UNK = ["<BEGIN>", "<END>", "<UNK>"]
PAD_TOKEN = -1
GPU_INDEX = 1
MAX_LENGTH = 500


class CCodeDataSet(Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary: Vocabulary,
                 transform=None):
        self.data_df = data_df[data_df['tokens'].map(lambda x: x is not None)]
        self.data_df = self.data_df[self.data_df['tokens'].map(lambda x: len(x) < MAX_LENGTH)]
        self.transform = transform
        self.vocabulary = vocabulary

        self._samples = [self._get_raw_sample(i) for i in range(len(self.data_df))]
        if self.transform:
            self._samples = show_process_map(self.transform, self._samples)
        # for s in self._samples:
        #     for k, v in s.items():
        #         print("{}:shape {}".format(k, np.array(v).shape))

    def _get_raw_sample(self, index):
        index_located_tuple = self.data_df.iloc[index]
        tokens = self.vocabulary.parse_text_without_pad([[k.value for k in index_located_tuple["tokens"]]],
                                                        use_position_label=True)[0]
        sample = {"tree": index_located_tuple["parse_tree"],
                  "tokens": tokens[:-1],
                  "target": tokens[1:],
                  "consistent_identifier": index_located_tuple['consistent_identifier'],
                  "identifier_scope_index": index_located_tuple['identifier_scope_index'],
                  "is_identifier": index_located_tuple['is_identifier'],
                  'max_scope_list': index_located_tuple['max_scope_list'],
                  'consistent_typename': index_located_tuple['consistent_typename'],
                  "length": len(tokens)-1}
        return sample

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)

class RangeMaskMap(object):
    def __init__(self, size):
        self._size = size

    def __call__(self, sample: int):
        if sample > self._size:
            raise ValueError("The range mask out of range, with size {} and sample {}".format(self._size, sample))
        return generate_mask(range(sample), self._size)

class GrammarLanguageModelTypeInputMap(object):
    """
    Map the top down parsing order node to the input format of the GrammarLanguageModel
    """
    def __init__(self,
                 production_vocabulary: SLKProductionVocabulary,
                 token_vocabulary: Vocabulary,
                 label_vocabulary: C99LabelVocabulary):
        self._production_vocabulary = production_vocabulary
        self._token_vocabulary = token_vocabulary
        self._label_vocabulary = label_vocabulary

    def _generate_terminal_mask(self, terminal_label_index, consistent_identifier, consistent_typename):
        size = self._token_vocabulary.vocabulary_size
        token_index_set = set()
        keyword_map = pre_defined_c_tokens_map
        for t in terminal_label_index:
            token_str = self._label_vocabulary.get_label_by_id(t)
            if token_str == "ID":
                for idt in consistent_identifier:
                    token_index_set.add(idt)
            elif token_str == "TYPEID":
                for idt in consistent_typename:
                    token_index_set.add(idt)
            elif token_str == "IMAGINARY_" or token_str == "END_OF_SLK_INPUT":
                pass
            else:
                token_index_set.add(self._token_vocabulary.word_to_id(keyword_map[token_str]))
        return generate_mask(token_index_set, size)


    def __call__(self, sample):
        """
        :param sample: a dict {"tree": a list of node, "consistent_identifier": consistent identifier string list,
        "consistent_typename": consistent typename string}
        :return: a dict of list {'to_parse_token', 'terminal_mask'}
        """
        # print()
        consistent_identifier = sample["consistent_identifier"]
        consistent_typename = sample["consistent_typename"]
        sample = sample["tree"]

        production_vocabulary = self._production_vocabulary

        get_matched_terminal_index = production_vocabulary.get_matched_terminal_node
        vocabulary_size = production_vocabulary.token_num()
        generate_mask_fn = generate_mask(size=vocabulary_size)
        get_node_right_id = lambda x: x.right_id

        stack = [production_vocabulary.EMPTY_id, sample[0].left_id]
        string_stack = ["EMPTY", sample[0].left]
        to_parse_token_id = [sample[0].left_id]
        now_id = 0
        peeked_max_id = -1

        sample = list(filter(lambda x: not(isinstance(x, LeafToken) and not production_vocabulary.is_token(x.type_id)),
                             sample))

        tokens = []
        for node in sample:
            if isinstance(node, LeafToken) and production_vocabulary.is_token(node.type_id):
                tokens.append(node.type_id)

        peeked_compact_dict = {}

        for node in sample:
            # print(node)
            type_id = stack.pop()
            type_string = string_stack.pop()
            # print("The stack popped token is:{}, string:{}".format(type_id, type_string))
            if isinstance(node, LeafToken):
                # print("Terminal token:{}".format(node.value))
                if production_vocabulary.is_token(node.type_id):
                    now_id +=1
                    to_parse_token_id.append(stack[-1])
            else:
                assert type_id == node.left_id, "type string is {}, now left is {}".format(type_string, node.left)
                if now_id < len(tokens) and production_vocabulary.need_peek(type_id, tokens[now_id]):
                    # print("need peek")
                    level = 1
                    entry = production_vocabulary.get_parse_entry(type_id, tokens[now_id])
                    peeked_id = now_id + level
                    if peeked_id not in peeked_compact_dict:
                        peeked_compact_dict[peeked_id] = production_vocabulary.get_conflict_matched_terminal_node(entry)
                    while production_vocabulary.need_peek(entry, tokens[peeked_id], True):
                        entry = production_vocabulary.get_conflict_entry(entry, tokens[peeked_id])
                        peeked_id += 1
                        if peeked_id not in peeked_compact_dict:
                            peeked_compact_dict[peeked_id] = production_vocabulary.get_conflict_matched_terminal_node(
                                entry)
                    peeked_max_id = max(peeked_max_id, peeked_id)

                for i, right_id in reversed(list(enumerate(get_node_right_id(node)))):
                    if production_vocabulary.is_token(right_id):
                        stack.append(right_id)
                        string_stack.append(node.right[i])
                    else:
                        # print("{} with id {} is not a token".format(node.right[i], right_id))
                        pass

        terminal_mask_index = []
        for i, token in enumerate(to_parse_token_id):
            if i in peeked_compact_dict:
                # print("peek", peeked_compact_dict[i])
                terminal_mask_index.append(peeked_compact_dict[i])
            else:
                # print("terminal", get_matched_terminal_index(token))
                terminal_mask_index.append(get_matched_terminal_index(token))

        terminal_mask = [self._generate_terminal_mask(index, a, b) for index, a, b in
                         zip(terminal_mask_index, consistent_identifier, consistent_typename)]

        return {"terminal_mask": terminal_mask,}

class PadMap(object):
    def __init__(self, terminal_num, scope_stack_size):
        self._terminal_pad = [0] * terminal_num
        self._scope_mask_pad = [0] * scope_stack_size

    def __call__(self, sample: dict):
        def pad_one_sample(x):
            x['tokens'] = padded_to_length(x['tokens'], MAX_LENGTH, 0)
            x['is_identifier'] = padded_to_length(x['tokens'], MAX_LENGTH, 0)
            x['identifier_scope_index'] = padded_to_length(x['identifier_scope_index'], MAX_LENGTH, 0)
            x['terminal_mask'] = padded_to_length(x['terminal_mask'], MAX_LENGTH, self._terminal_pad)
            x['max_scope_list'] = padded_to_length(x['terminal_mask'], MAX_LENGTH, self._scope_mask_pad)
            x['target'] = padded_to_length(x['target'], MAX_LENGTH, PAD_TOKEN)
            return x
        return pad_one_sample(sample)

class ScopeGrammarLanguageModel(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_state_size,
                 rnn_num_layers,
                 stack_size,
                 batch_size):
        super().__init__()
        self._batch_size = batch_size
        self._rnn_num_layers = rnn_num_layers
        self._hidden_state_size = hidden_state_size
        self._stack_size = stack_size

        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.rnn = nn.GRUCell(input_size=embedding_dim,
                           hidden_size=hidden_state_size,
                           num_layers=rnn_num_layers,).cuda(GPU_INDEX)
        self.scope_transformer = nn.Sequential(
            nn.Linear(embedding_dim+hidden_state_size, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, embedding_dim)).cuda(GPU_INDEX)
        self._initial_state = self.initial_state()

    def initial_state(self):
        return (nn.Parameter(torch.randn((self._rnn_num_layers, self._batch_size, self._hidden_state_size)),
                             requires_grad=True).cuda(GPU_INDEX),
                nn.Parameter(torch.randn((self._rnn_num_layers, self._batch_size, self._hidden_state_size)),
                             requires_grad=True).cuda(GPU_INDEX))

    def _embedding(self, token_sequence):
        """
        :param token_sequence: a long variable with the shape [batch, seq]
        :return: The embedding seq of shape [batch, seq, embedding_dim]
        """
        return self.token_embeddings(token_sequence).cuda(GPU_INDEX)

    def _forward_rnn(self,
                     tokens,
                     identifier_scope_index,
                     lengths):
        """
        :param tokens: a float variable with the shape [batch, seq,]
        :param lengths: a long variable with the shape [batch, ]
        :return: a float variable with the shape [batch, seq, feature]
        """
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(tokens, lengths, batch_first=True)
        identifier_scope_index = torch.nn.utils.rnn.pack_padded_sequence(identifier_scope_index, lengths,
                                                                         batch_first=True)

        # print("packed seq size:{}".format(packed_seq.data.data.size()))
        packed_seq = torch.nn.utils.rnn.PackedSequence(self._embedding(packed_seq.data), packed_seq.batch_sizes)
        output, _ = self.rnn(packed_seq, self._initial_state)
        return output

    def new_begin_stack(self):
        return torch.zeros((self._batch_size, self._stack_size, self._hidden_state_size)).cuda(GPU_INDEX)

    def forward(self, tokens, identifier_scope_index):
        pass