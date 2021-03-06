import sys
import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorflow.python.util.nest import map_structure
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import pandas as pd

import os

import config
from c_code_processer.code_util import LeafToken
from c_code_processer.fake_c_header.extract_identifier import extract_fake_c_header_identifier
from c_code_processer.slk_parser import SLKProductionVocabulary, C99LabelVocabulary, C99SLKConstants
from common import torch_util, util
from common.constants import pre_defined_c_tokens_map, CACHE_DATA_PATH
from common.torch_util import calculate_accuracy_of_code_completion
from common.util import show_process_map, generate_mask, padded_to_length, key_transform, FlatMap, data_loader, CopyMap, \
    disk_cache, inplace_show_process_map
from embedding.wordembedding import Vocabulary, load_vocabulary, load_keyword_identifier_split_vocabulary
from read_data.load_parsed_data import get_token_vocabulary, get_vocabulary_id_map_with_keyword, \
    read_monitored_parsed_c99_slk_top_down_code, load_positioned_keyword_identifier_split_vocabulary, \
    read_monitored_parsed_c99_slk_top_down_code_without_consistent_name

BEGIN, END, UNK = ["<BEGIN>", "<END>", "<UNK>"]
PAD_TOKEN = -1
GPU_INDEX = 0
MAX_LENGTH = 500
IDENTIFIER_BEGIN_INDEX = 84


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
        del self.data_df

        def error_filter(sample):
            try:
                self.transform(sample)
                return True
            except Exception as e:
                print(e)
                return False
        self._samples = list(filter(error_filter, self._samples))

        # for s in self._samples:
        #     for k, v in s.items():
        #         print("{}:shape {}".format(k, np.array(v).shape))

    def _get_raw_sample(self, index):
        index_located_tuple = self.data_df.iloc[index]
        tokens = self.vocabulary.parse_text_without_pad([[k for k in index_located_tuple["tokens"]]],
                                                        use_position_label=True)[0]
        sample = {"tree": index_located_tuple["parse_tree"],
                  "tokens": tokens[:-1],
                  "target": tokens[1:],
                  # "consistent_identifier": index_located_tuple['consistent_identifier'],
                  # 'consistent_typename': index_located_tuple['consistent_typename'],
                  "length": len(tokens) - 1}
        return sample

    # def transform_fn(self):
    #     self._samples = inplace_show_process_map(self.transform, self._samples, error_default_value=None)
    #     self._samples = list(filter(lambda x: x is not None, self._samples))

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self._samples[index])
        else:
            return self._samples[index]

    def __len__(self):
        return len(self._samples)


class RangeMaskMap(object):
    def __init__(self, size):
        self._size = size

    def _g_map(self, sample: int):
        if sample > self._size:
            raise ValueError("The range mask out of range, with size {} and sample {}".format(self._size, sample))
        return generate_mask(range(sample), self._size)

    def __call__(self, sample):
        return [self._g_map(t) for t in sample]


class GrammarLanguageModelTypeInputMap(object):
    """
    Map the top down parsing order node to the input format of the GrammarLanguageModel
    """

    def __init__(self,
                 production_vocabulary: SLKProductionVocabulary,
                 token_vocabulary: Vocabulary,
                 label_vocabulary: C99LabelVocabulary,
                 keyword_num):
        self._production_vocabulary = production_vocabulary
        self._token_vocabulary = token_vocabulary
        self._label_vocabulary = label_vocabulary
        self._c_header_identifier, self._c_header_type = extract_fake_c_header_identifier()
        self._c_header_identifier = {token_vocabulary.word_to_id(t) for t in self._c_header_identifier}
        self._c_header_type = {token_vocabulary.word_to_id(t) for t in self._c_header_type}
        keyword_map = pre_defined_c_tokens_map
        self._identifier_index = (IDENTIFIER_BEGIN_INDEX, token_vocabulary.vocabulary_size-1)
        self._keyword_num = keyword_num

    def _generate_terminal_mask(self, terminal_label_index,):
        size = self._keyword_num
        token_index_set = set()
        keyword_map = pre_defined_c_tokens_map
        has_identifer = 0
        for t in terminal_label_index:
            token_str = self._label_vocabulary.get_label_by_id(t)
            if token_str == "ID":
                has_identifer = 1
            elif token_str == "TYPEID":
                has_identifer = 1
            elif token_str == "IMAGINARY_":
                pass
            elif token_str == "END_OF_SLK_INPUT":
                token_index_set.add(self._token_vocabulary.word_to_id(END))
            elif token_str == "CONSTANT" or token_str == "STRING_LITERAL":
                token_index_set.add(self._token_vocabulary.word_to_id(token_str))
            else:
                token_index_set.add(self._token_vocabulary.word_to_id(keyword_map[token_str]))
        return generate_mask(token_index_set, size).flip(), 1 - has_identifer

    def __call__(self, sample):
        """
        :param sample: a dict {"tree": a list of node, }
        :return: a dict of list {'to_parse_token', 'terminal_mask'}
        """
        # print()
        # consistent_identifier = sample["consistent_identifier"]
        # consistent_typename = sample["consistent_typename"]
        target = sample['target']
        target_string = [self._token_vocabulary.id_to_word(t) for t in target]
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

        sample = list(filter(lambda x: not (isinstance(x, LeafToken) and not production_vocabulary.is_token(x.type_id)),
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
                    now_id += 1
                    to_parse_token_id.append(stack[-1])
            else:
                assert type_id == node.left_id, "type string is {}, now left is {}".format(type_string, node.left)
                # print(node.left)
                if now_id < len(tokens) and production_vocabulary.need_peek(type_id, tokens[now_id]):
                    # print("need peek")
                    level = 1
                    entry = production_vocabulary.get_parse_entry(type_id, tokens[now_id])
                    # print("entry is:{}".format(entry))
                    peeked_id = now_id + level
                    if peeked_id not in peeked_compact_dict:
                        # print("token {} need peek after token {} saw".format(target_string[peeked_id],  target_string[now_id]))
                        peeked_compact_dict[peeked_id] = production_vocabulary.get_conflict_matched_terminal_node(entry)
                        # print("token {} in peeked_dict? {}".format(target_string[peeked_id], tokens[peeked_id] in peeked_compact_dict[peeked_id]))
                    while production_vocabulary.need_peek(entry, tokens[peeked_id], True):
                        entry = production_vocabulary.get_conflict_entry(entry, tokens[peeked_id])
                        # print("now entry:{}".format(entry))
                        peeked_id += 1
                        # print("token {} need peek after token {} saw".format(target_string[peeked_id],  target_string[now_id]))
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
        for (i, token), t in zip(enumerate(to_parse_token_id), target):
            if i in peeked_compact_dict:
                # print("peek", peeked_compact_dict[i])
                # print("target {} use peek".format(self._token_vocabulary.id_to_word(t)))
                terminal_mask_index.append(peeked_compact_dict[i])
            else:
                # print("terminal", get_matched_terminal_index(token))
                # print("target {} use get matched".format(self._token_vocabulary.id_to_word(t)))
                terminal_mask_index.append(get_matched_terminal_index(token))

        terminal_mask = [self._generate_terminal_mask(index) for index in
                         terminal_mask_index ]
        from toolz.sandbox import unzip
        terminal_mask, has_identifier = unzip(terminal_mask)
        terminal_mask = list(terminal_mask)
        for t in terminal_mask:
            assert len(t) == self._keyword_num
        has_identifier = list(has_identifier)

        prev_tokens = []
        for t, mask, index, h_i in zip(target, terminal_mask, terminal_mask_index, has_identifier):
            if t < self._keyword_num:
                if mask[t] != 0:
                    # print("The code before: {}".format(" ".join([self._token_vocabulary.id_to_word(to) for to in prev_tokens])))
                    # print("all code:{}".format(" ".join([self._token_vocabulary.id_to_word(to) for to in target])))
                    msg = "target {} not in the mask".format(self._token_vocabulary.id_to_word(t))
                    raise ValueError(msg)
            elif h_i != 0:
                # print("The code before: {}".format(
                #     " ".join([self._token_vocabulary.id_to_word(to) for to in prev_tokens])))
                # print("all code:{}".format(" ".join([self._token_vocabulary.id_to_word(to) for to in target])))
                msg = "target {} not in the mask".format(self._token_vocabulary.id_to_word(t))
                raise ValueError(msg)
            else:
                prev_tokens.append(t)

        return {"terminal_mask": terminal_mask, "target": target, "has_identifier": has_identifier}


class IndexMaskMap(object):
    def __init__(self, size):
        self._size = size

    def __call__(self, sample):
        return [generate_mask([t], self._size) for t in sample]


class PadMap(object):
    def __init__(self, terminal_num):
        self._terminal_pad = [0] * terminal_num

    def __call__(self, sample: dict):
        def pad_one_sample(x):
            x['tokens'] = padded_to_length(x['tokens'], MAX_LENGTH, 0)
            x['has_identifier'] = padded_to_length(x['has_identifier'], MAX_LENGTH, 0)

            x['terminal_mask'] = padded_to_length(x['terminal_mask'], MAX_LENGTH, self._terminal_pad)
            x['target'] = padded_to_length(x['target'], MAX_LENGTH, PAD_TOKEN)
            return x

        return pad_one_sample(sample)


class ScopeGrammarLanguageModel(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_state_size,
                 rnn_num_layers,
                 keyword_size,
                 batch_size):
        super().__init__()
        self._batch_size = batch_size
        self._rnn_num_layers = rnn_num_layers
        self._hidden_state_size = hidden_state_size

        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.rnn = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_state_size,
                           num_layers=rnn_num_layers,).cuda(GPU_INDEX)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, vocab_size-keyword_size)).cuda(GPU_INDEX)
        self.output_keyword_mlp = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, keyword_size)).cuda(GPU_INDEX)
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
                     lengths):
        """
        :param tokens: a float variable with the shape [batch, seq,]
        :param lengths: a long variable with the shape [batch, ]
        :return: a float variable with the shape [batch, seq, feature]
        """
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(tokens, lengths, batch_first=True)
        # print("packed seq size:{}".format(packed_seq.data.data.size()))
        packed_seq = torch.nn.utils.rnn.PackedSequence(self._embedding(packed_seq.data), packed_seq.batch_sizes)
        output, _ = self.rnn(packed_seq, self._initial_state)
        return output.data

    def forward(self,
                tokens,
                terminal_mask,
                lengths,
                has_identifier):
        rnn_feature = self._forward_rnn(
            tokens,
            lengths
        )
        terminal_mask = torch.nn.utils.rnn.pack_padded_sequence(terminal_mask, lengths, batch_first=True)
        has_identifier = torch.nn.utils.rnn.pack_padded_sequence(has_identifier, lengths, batch_first=True).data.cuda(
            GPU_INDEX)
        batch_sizes = terminal_mask.batch_sizes
        terminal_mask = terminal_mask.data.cuda(GPU_INDEX)
        predict = self.output_keyword_mlp(rnn_feature)
        predict.data.masked_fill_(terminal_mask, -float('inf'))
        identifier_predict = self.output_mlp(rnn_feature)
        identifier_predict.masked_fill_(has_identifier.unsqueeze(1), -float('inf'))
        predict = torch.cat((predict, identifier_predict), dim=1)
        predict = torch.nn.utils.rnn.PackedSequence(predict, batch_sizes)
        predict, _ = torch.nn.utils.rnn.pad_packed_sequence(predict, batch_first=True, padding_value=PAD_TOKEN)
        return predict


def train(model,
          dataset,
          batch_size,
          loss_function,
          optimizer):
    total_loss = torch.Tensor([0]).cuda(GPU_INDEX)
    steps = torch.Tensor([0]).cuda(GPU_INDEX)
    for batch_data in data_loader(dataset, batch_size=batch_size, is_shuffle=True, drop_last=True, epoch_ratio=0.5):
        # print(batch_data['terminal_mask'])
        # print('batch_data size: ', len(batch_data['terminal_mask'][0]), len(batch_data['terminal_mask'][0][0]))
        # res = list(more_itertools.collapse(batch_data['terminal_mask']))
        # print('res len: ', len(res))
        # res = util.padded(batch_data['terminal_mask'], deepcopy=True, fill_value=0)
        # print('batch_data size: ', len(res[0]), len(res[0][0]))
        # res = list(more_itertools.collapse(res))
        # print('res len: ', len(res))
        lengths, target, terminal_mask, tokens, has_identifier\
            = parse_batch_data(
            batch_data)
        # print("parsed data")
        model.zero_grad()
        log_probs = model.forward(
            tokens,
            terminal_mask,
            lengths, has_identifier
        )
        # log_probs.register_hook(create_hook_fn("log_probs"))

        # print("log_probs sizze:{}".format(log_probs.size()))
        batch_log_probs = log_probs.contiguous().view(-1, list(log_probs.size())[-1])

        target, idx_unsort = torch_util.pack_padded_sequence(
            autograd.Variable(torch.LongTensor(target)).cuda(GPU_INDEX),
            lengths, batch_firse=True, GPU_INDEX=GPU_INDEX)
        target, _ = torch_util.pad_packed_sequence(target, idx_unsort, pad_value=PAD_TOKEN, batch_firse=True,
                                                   GPU_INDEX=GPU_INDEX)

        loss = loss_function(batch_log_probs, target.view(-1))

        # loss.register_hook(create_hook_fn("loss"))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        # print()
        # print("The loss is nan:{}".format(is_nan(loss.detach())))
        # print("The loss grad is nan:{}".format(is_nan(loss.grad)))
        # print("The log_probs is nan:{}".format(is_nan(log_probs.detach())))
        # print("The log_probs grad is nan:{}".format(is_nan(log_probs.grad)))
        # for name, param in model.named_parameters():
        #     print("name of {}: has nan:{}".format(name, is_nan(param.detach())))
        #     print("the gradient of {}: has nan:{}".format(name, is_nan(param.grad)))
        # if HAS_NAN:
        #     for k, v in batch_data.items():
        #         print("{}:{}".format(k, show_tensor(v)))
        #     print("{}:{}".format("target", show_tensor(target)))
        # print()

        optimizer.step()

        # print("loss：{}".format(loss.data))
        total_loss += loss.data
        steps += torch.sum(lengths.data)
    return total_loss / steps


def parse_batch_data(batch_data):
    lengths = batch_data['length']
    _, idx_sort = torch.sort(torch.LongTensor(lengths), dim=0, descending=True)
    batch_data = {k: util.index_select(v, idx_sort) for k, v in batch_data.items()}
    tokens = autograd.Variable(torch.LongTensor(batch_data['tokens']))
    terminal_mask = autograd.Variable(torch.ByteTensor(batch_data['terminal_mask']))
    # print("update_mask size:{}".format(update_mask.size()))
    lengths = autograd.Variable(torch.LongTensor(batch_data['length']))
    has_identifier = autograd.Variable(torch.ByteTensor(batch_data['has_identifier']))
    target = batch_data["target"]
    return lengths, target, terminal_mask, tokens, has_identifier


def evaluate(model,
             dataset,
             batch_size,
             loss_function):
    total_loss = torch.Tensor([0]).cuda(GPU_INDEX)
    steps = torch.Tensor([0]).cuda(GPU_INDEX)
    for batch_data in data_loader(dataset, batch_size=batch_size, is_shuffle=True, drop_last=True):
        lengths, target, terminal_mask, tokens, has_identifier\
            = parse_batch_data(
            batch_data)
        log_probs = model.forward(tokens,
                                  terminal_mask,
                                  lengths, has_identifier)

        batch_log_probs = log_probs.contiguous().view(-1, list(log_probs.size())[-1])

        target, idx_unsort = torch_util.pack_padded_sequence(
            autograd.Variable(torch.LongTensor(target)).cuda(GPU_INDEX),
            lengths, batch_firse=True, GPU_INDEX=GPU_INDEX)
        target, _ = torch_util.pad_packed_sequence(target, idx_unsort, pad_value=PAD_TOKEN, batch_firse=True,
                                                   GPU_INDEX=GPU_INDEX)

        loss = loss_function(batch_log_probs, target.view(-1))
        total_loss += loss.data
        steps += torch.sum(lengths.data)
    return total_loss / steps


def train_and_evaluate(data,
                       keyword_num,
                       vocabulary,
                       batch_size,
                       embedding_dim,
                       hidden_state_size,
                       rnn_num_layer,
                       learning_rate,
                       epoches,
                       saved_name,
                       load_previous_model=False):
    save_path = os.path.join(config.save_model_root, saved_name)
    # for d in data:
    #     def get_i(i):
    #         return d[i]
    #     show_process_map(get_i, range(len(d)))
    for d, n in zip(data, ["train", "val", "test"]):
        print("There are {} parsed data in the {} dataset".format(len(d), n))
    train_dataset, valid_dataset, test_dataset = data

    loss_function = nn.CrossEntropyLoss(size_average=False, ignore_index=PAD_TOKEN)
    model = ScopeGrammarLanguageModel(
        vocabulary.vocabulary_size,
        embedding_dim,
        hidden_state_size,
        rnn_num_layer,
        keyword_num,
        batch_size,
    )
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    if load_previous_model:
        torch_util.load_model(model, save_path)
        valid_loss = evaluate(model, valid_dataset, batch_size, loss_function)
        test_loss = evaluate(model, test_dataset, batch_size, loss_function)
        best_valid_perplexity = torch.exp(valid_loss).item()
        best_test_perplexity = torch.exp(test_loss).item()
        print(
            "load the previous mode, validation perplexity is {}, test perplexity is :{}".format(best_valid_perplexity,
                                                                                                 best_test_perplexity))
        scheduler.step(best_valid_perplexity)
    else:
        best_valid_perplexity = None
        best_test_perplexity = None
    begin_time = time.time()
    # with torch.autograd.profiler.profile() as prof:
    for epoch in range(epoches):
        train_loss = train(model, train_dataset, batch_size, loss_function, optimizer)
        valid_loss = evaluate(model, valid_dataset, batch_size, loss_function)
        test_loss = evaluate(model, test_dataset, batch_size, loss_function)

        train_perplexity = torch.exp(train_loss).item()
        valid_perplexity = torch.exp(valid_loss).item()
        test_perplexity = torch.exp(test_loss).item()

        scheduler.step(valid_perplexity)

        if best_valid_perplexity is None or valid_perplexity < best_valid_perplexity:
            best_valid_perplexity = valid_perplexity
            best_test_perplexity = test_perplexity
            torch_util.save_model(model, save_path)

        print("epoch {}: train perplexity of {},  valid perplexity of {}, test perplexity of {}".
              format(epoch, train_perplexity, valid_perplexity, test_perplexity))
    # print(prof)
    print("The model {} best valid perplexity is {} and test perplexity is {}".
          format(saved_name, best_valid_perplexity, best_test_perplexity))
    print("use time {} seconds".format(time.time() - begin_time))


def transform_data_from_df_to_dataset(data,):
    for d, n in zip(data, ["train", "val", "test"]):
        print("There are {} raw data in the {} dataset".format(len(d), n))
    vocabulary, keyword_num = load_keyword_identifier_split_vocabulary(get_token_vocabulary, [BEGIN], [END], UNK)
    print("vocab_size:{}".format(vocabulary.vocabulary_size))
    print("The max token id:{}".format(max(vocabulary.word_to_id_dict.values())))
    slk_constants = C99SLKConstants()
    # terminal_token_index = set(range(slk_constants.START_SYMBOL-2)) - {63, 64}
    label_vocabulary = C99LabelVocabulary(slk_constants)
    production_vocabulary = SLKProductionVocabulary(slk_constants)
    transforms_fn = transforms.Compose([
        # IsNone("original"),
        CopyMap(),
        key_transform(
            GrammarLanguageModelTypeInputMap(production_vocabulary, vocabulary, label_vocabulary, keyword_num),
            "tree", "target"),
        # IsNone("after type input"),
        FlatMap(),
        # IsNone("Flat Map"),
        PadMap(keyword_num),
        # IsNone("Pad Map"),
    ])
    generate_dataset = lambda df: CCodeDataSet(df, vocabulary, transforms_fn)
    res = generate_dataset(data[0])
    del data[0]
    return res, keyword_num, vocabulary


# @disk_cache(basename="grammar_baseline_language_model_load_parsed_data", directory=CACHE_DATA_PATH)
def load_parsed_data():
    data = read_monitored_parsed_c99_slk_top_down_code_without_consistent_name()
    for d, n in zip(data, ["train", "val", "test"]):
        print("There are {} raw data in the {} dataset".format(len(d), n))
    res = [0] * 3
    res[0], keyword_num, vocabulary = transform_data_from_df_to_dataset(data)
    res[1], keyword_num, vocabulary = transform_data_from_df_to_dataset(data)
    res[2], keyword_num, vocabulary = transform_data_from_df_to_dataset(data)
    return res, keyword_num, vocabulary


def accuracy_evaluate(model,
                      dataset,
                      batch_size,
                      loss_function,):
    total_loss = torch.Tensor([0]).cuda(GPU_INDEX)
    steps = torch.Tensor([0]).cuda(GPU_INDEX)
    accuracy_dict = None
    for batch_data in data_loader(dataset, batch_size=batch_size, is_shuffle=True, drop_last=True):
        lengths, target, terminal_mask, tokens, has_identifier\
            = parse_batch_data(
            batch_data)
        log_probs = model.forward(tokens,
                                  terminal_mask,
                                  lengths, has_identifier)

        batch_log_probs = log_probs.contiguous().view(-1, list(log_probs.size())[-1])

        target, idx_unsort = torch_util.pack_padded_sequence(
            autograd.Variable(torch.LongTensor(target)).cuda(GPU_INDEX),
            lengths, batch_firse=True, GPU_INDEX=GPU_INDEX)
        target, _ = torch_util.pad_packed_sequence(target, idx_unsort, pad_value=PAD_TOKEN, batch_firse=True,
                                                   GPU_INDEX=GPU_INDEX)

        loss = loss_function(batch_log_probs, target.view(-1))
        total_loss += loss.data
        steps += torch.sum(lengths.data)
        # print("target size:{}".format(target.size()))
        # print("batch log size:{}".format(log_probs.size()))
        topk_accuracy = calculate_accuracy_of_code_completion(log_probs, target, ignore_token=PAD_TOKEN, gpu_index=GPU_INDEX)
        if accuracy_dict is None:
            accuracy_dict = topk_accuracy
        else:
            for k, v in topk_accuracy.items():
                accuracy_dict[k] += topk_accuracy[k]
    accuracy_dict = {k: float(v)/steps.item() for k, v in accuracy_dict.items()}
    return total_loss / steps, accuracy_dict


def load_test_data():
    # cache_path = os.path.join(CACHE_DATA_PATH, "scope_grammar_language_model_parsed_test_data.pkl")
    # print("cached_path:{}".format(cache_path))
    # if os.path.isfile(cache_path):
    #     with open(cache_path, 'rb') as handle:
    #         print("load cache from:{}".format(cache_path))
    #         res, keyword_num, vocabulary = pickle.load(handle)
    # else:
    data = read_monitored_parsed_c99_slk_top_down_code_without_consistent_name()
    for d, n in zip(data, ["train", "val", "test"]):
        print("There are {} raw data in the {} dataset".format(len(d), n))
    data = data[-1:]
    res, keyword_num, vocabulary = transform_data_from_df_to_dataset(data,)
        # with open(cache_path, 'wb') as handle:
        #     print("dump cache from:{}".format(cache_path))
        #     pickle.dump([res, keyword_num, vocabulary], handle)
    print("parsed test data size:{}".format(len(res)))
    return res, keyword_num, vocabulary


def only_evaluate(data,
                  keyword_num,
                  vocabulary,
                  batch_size,
                  embedding_dim,
                  hidden_state_size,
                  rnn_num_layer,
                  learning_rate,
                  epoches,
                  saved_name,
                  stack_size,
                  load_previous_model=False):
    save_path = os.path.join(config.save_model_root, saved_name)
    # for d in data:
    #     def get_i(i):
    #         return d[i]
    #     show_process_map(get_i, range(len(d)))
    # for d, n in zip(data, ["train", "val", "test"]):
    #     print("There are {} parsed data in the {} dataset".format(len(d), n))
    test_dataset = data

    loss_function = nn.CrossEntropyLoss(size_average=False, ignore_index=PAD_TOKEN)
    model = ScopeGrammarLanguageModel(
        vocabulary.vocabulary_size,
        embedding_dim,
        hidden_state_size,
        rnn_num_layer,
        keyword_num,
        batch_size,
    )
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    torch_util.load_model(model, save_path)
    test_loss, top_k_accuracy = accuracy_evaluate(model, test_dataset, batch_size, loss_function)
    best_test_perplexity = torch.exp(test_loss).item()
    print(
        "load the previous mode, test perplexity is :{}".format(
                                                                                             best_test_perplexity))
    # print(prof)
    print("The model {} test perplexity is {}".
          format(saved_name, best_test_perplexity))
    print("The top k accuracy:")
    for k, v in top_k_accuracy.items():
        print("{}：{}".format(k, v))

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    data, keyword_num, vocabulary = load_test_data()
    # only_evaluate(data, keyword_num, vocabulary, 4, 100, 100, 3, 0.01, 50, "grammar_baseline_language_model_1.pkl", None,
    #               load_previous_model=True)
    # data, keyword_num, vocabulary = load_parsed_data()
    # print(data[0]['code'][0])
    # train_and_evaluate(data, keyword_num, vocabulary, 16, 100, 100, 3, 0.01, 50, "grammar_baseline_language_model_1.pkl",
    #                    load_previous_model=False)
    # The model c89_grammar_lm_1.pkl best valid perplexity is 2.7838220596313477 and test perplexity is 2.7718544006347656
    # train_and_evaluate(data, keyword_num, vocabulary, 16, 200, 200, 3, 0.01, 50, "grammar_baseline_language_model_2.pkl",
    #                    load_previous_model=False)
    # The model c89_grammar_lm_2.pkl best valid perplexity is 3.062429189682007 and test perplexity is 3.045041799545288
    # train_and_evaluate(data, keyword_num, vocabulary, 16, 300, 300, 3, 0.01, 50, "grammar_baseline_language_model_3.pkl",
    #                    load_previous_model = False)
    # The model c89_grammar_lm_3.pkl best valid perplexity is 2.888122797012329 and test perplexity is 2.8750290870666504
