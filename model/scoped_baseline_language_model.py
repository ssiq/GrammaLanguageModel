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
                 stack_size,
                 transform=None):
        self.data_df = data_df[data_df['tokens'].map(lambda x: x is not None)]
        self.data_df = self.data_df[self.data_df['tokens'].map(lambda x: len(x) < MAX_LENGTH)]
        self.transform = transform
        self.vocabulary = vocabulary

        self._samples = [self._get_raw_sample(i) for i in range(len(self.data_df))]
        del self.data_df
        del data_df
        self._samples = list(filter(lambda x: max(x['max_scope_list']) <= stack_size, self._samples))

        # for s in self._samples:
        #     for k, v in s.items():
        #         print("{}:shape {}".format(k, np.array(v).shape))

    def _get_raw_sample(self, index):
        index_located_tuple = self.data_df.iloc[index]
        tokens = self.vocabulary.parse_text_without_pad([[k for k in index_located_tuple["tokens"]]],
                                                        use_position_label=True)[0]
        sample = {"tokens": tokens[:-1],
                  "target": tokens[1:],
                  # "consistent_identifier": index_located_tuple['consistent_identifier'],
                  "identifier_scope_index": index_located_tuple['identifier_scope_index'],
                  "is_identifier": index_located_tuple['is_identifier'],
                  'max_scope_list': index_located_tuple['max_scope_list'],
                  # 'consistent_typename': index_located_tuple['consistent_typename'],
                  "length": len(tokens) - 1}
        return sample

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


class IndexMaskMap(object):
    def __init__(self, size):
        self._size = size

    def __call__(self, sample):
        return [generate_mask([t], self._size) for t in sample]


class PadMap(object):
    def __init__(self, terminal_num, scope_stack_size):
        self._terminal_pad = [0] * terminal_num
        self._scope_mask_pad = [0] * scope_stack_size

    def __call__(self, sample: dict):
        def pad_one_sample(x):
            x['tokens'] = padded_to_length(x['tokens'], MAX_LENGTH, 0)
            x['is_identifier'] = padded_to_length(x['is_identifier'], MAX_LENGTH, 0)
            x['identifier_scope_index'] = padded_to_length(x['identifier_scope_index'], MAX_LENGTH,
                                                           self._scope_mask_pad)
            x['max_scope_list'] = padded_to_length(x['max_scope_list'], MAX_LENGTH, self._scope_mask_pad)
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
                 stack_size,
                 batch_size):
        super().__init__()
        self._batch_size = batch_size
        self._rnn_num_layers = rnn_num_layers
        self._hidden_state_size = hidden_state_size
        self._stack_size = stack_size

        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.rnn = torch_util.MultiRNNCell(
            [nn.LSTMCell(input_size=embedding_dim, hidden_size=hidden_state_size, )]
            + [nn.LSTMCell(input_size=hidden_state_size, hidden_size=hidden_state_size, ) for _ in
               range(rnn_num_layers - 1)]
        ).cuda(GPU_INDEX)
        self.scope_transformer = nn.Sequential(
            nn.Linear(embedding_dim + hidden_state_size, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, embedding_dim)).cuda(GPU_INDEX)
        self.scope_stack_update_gate = nn.Sequential(
            nn.Linear(hidden_state_size + hidden_state_size, 1),
            nn.Sigmoid()
        ).cuda(GPU_INDEX)
        self.scope_stack_update_value = nn.Sequential(
            nn.Linear(hidden_state_size + hidden_state_size, hidden_state_size),
            nn.Tanh()
        ).cuda(GPU_INDEX)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, vocab_size)).cuda(GPU_INDEX)
        self._initial_state = self.initial_state()
        self._begin_stack = self._new_begin_stack()

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

    def _scoped_input(self, now_identifier_scope_mask, now_token_embedding, now_is_identifier, scope_stack):
        now_scope = torch.masked_select(scope_stack, now_identifier_scope_mask.unsqueeze(2)).view(
            *now_token_embedding.size())
        # print("now scope size:{}".format(now_scope.size()))
        # print("now_token_embedding size:{}".format(now_token_embedding.size()))
        scoped_token_embedding = self.scope_transformer(torch.cat((now_scope, now_token_embedding), dim=1))
        return torch.where(now_is_identifier.unsqueeze(1), scoped_token_embedding, now_token_embedding)

    def _update_scope_stack(self, scope_stack, rnn_output, update_mask):
        rnn_output = torch.unsqueeze(rnn_output, 1, ).expand(*scope_stack.size())
        # print("scope_stack size:{}".format(scope_stack.size()))
        # print("rnn_output size:{}".format(rnn_output.size()))
        scope_update_input = torch.cat((scope_stack, rnn_output), dim=2)
        # print("scope_update_input size:{}".format(scope_update_input.size()))
        update_gate = self.scope_stack_update_gate(scope_update_input)
        update_value = self.scope_stack_update_value(scope_update_input)
        scope_stack = scope_stack * (1 - update_gate) + update_gate * update_value
        scope_stack.masked_fill_(update_mask.unsqueeze(2), 0)
        return scope_stack

    def _forward_rnn(self,
                     tokens,
                     identifier_scope_mask,
                     is_identifier,
                     update_mask,
                     lengths):
        """
        :param tokens: a float variable with the shape [batch, seq,]
        :param lengths: a long variable with the shape [batch, ]
        :return: a float variable with the shape [batch, seq, feature]
        """
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(tokens, lengths, batch_first=True)
        identifier_scope_mask = torch.nn.utils.rnn.pack_padded_sequence(identifier_scope_mask, lengths,
                                                                        batch_first=True).data.cuda(GPU_INDEX)
        is_identifier = torch.nn.utils.rnn.pack_padded_sequence(is_identifier, lengths,
                                                                batch_first=True).data.cuda(GPU_INDEX)
        update_mask = torch.nn.utils.rnn.pack_padded_sequence(update_mask, lengths,
                                                              batch_first=True).data.cuda(GPU_INDEX)

        # print("packed seq size:{}".format(packed_seq.data.data.size()))
        batch_sizes = packed_seq.batch_sizes
        packed_seq = self._embedding(packed_seq.data)
        now_state = self._initial_state
        now_scope_stack = self._begin_stack
        output = []
        begin_index = 0
        end_index = 0
        for i in range(len(batch_sizes)):
            end_index += batch_sizes[i]
            now_scope_stack = now_scope_stack[:end_index - begin_index]
            rnn_input = self._scoped_input(
                identifier_scope_mask[begin_index: end_index],
                packed_seq[begin_index:end_index],
                is_identifier[begin_index:end_index],
                now_scope_stack)
            now_state = map_structure(lambda x: x[:end_index - begin_index], now_state)
            now_o, now_state = self.rnn(rnn_input, now_state)
            output.append(now_o)
            now_scope_stack = self._update_scope_stack(
                now_scope_stack,
                now_o,
                update_mask[begin_index: end_index]
            )
            begin_index += batch_sizes[i]
        # print("output size:")
        # for t in output:
        #     print(t.size())
        return torch.cat(output, dim=0), batch_sizes

    def _new_begin_stack(self):
        return torch.zeros((self._batch_size, self._stack_size, self._hidden_state_size)).cuda(GPU_INDEX)

    def forward(self,
                tokens,
                identifier_scope_mask,
                is_identifier,
                update_mask,
                lengths):
        rnn_feature, batch_sizes = self._forward_rnn(
            tokens,
            identifier_scope_mask,
            is_identifier,
            update_mask,
            lengths
        )
        predict = self.output_mlp(rnn_feature)
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
        identifier_scope_mask, is_identifier, lengths, target, tokens, update_mask\
            = parse_batch_data(
            batch_data)
        # print("parsed data")
        model.zero_grad()
        log_probs = model.forward(
            tokens,
            identifier_scope_mask,
            is_identifier,
            update_mask,
            lengths
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
    is_identifier = autograd.Variable(torch.ByteTensor(batch_data['is_identifier']))
    identifier_scope_mask = autograd.Variable(torch.ByteTensor(batch_data['identifier_scope_index']))
    update_mask = autograd.Variable(torch.ByteTensor(batch_data['max_scope_list']))
    # print("update_mask size:{}".format(update_mask.size()))
    lengths = autograd.Variable(torch.LongTensor(batch_data['length']))
    target = batch_data["target"]
    return identifier_scope_mask, is_identifier, lengths, target, tokens, update_mask


def evaluate(model,
             dataset,
             batch_size,
             loss_function):
    total_loss = torch.Tensor([0]).cuda(GPU_INDEX)
    steps = torch.Tensor([0]).cuda(GPU_INDEX)
    for batch_data in data_loader(dataset, batch_size=batch_size, is_shuffle=True, drop_last=True):
        identifier_scope_mask, is_identifier, lengths, target, tokens, update_mask\
            = parse_batch_data(
            batch_data)
        log_probs = model.forward(tokens,
                                  identifier_scope_mask,
                                  is_identifier,
                                  update_mask,
                                  lengths)

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
                       stack_size,
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
        stack_size,
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


def transform_data_from_df_to_dataset(data, stack_size):
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
        key_transform(RangeMaskMap(stack_size), "max_scope_list"),
        key_transform(IndexMaskMap(stack_size), "identifier_scope_index"),
        # IsNone("after type input"),
        FlatMap(),
        # IsNone("Flat Map"),
        PadMap(keyword_num, stack_size),
        # IsNone("Pad Map"),
    ])
    generate_dataset = lambda df: CCodeDataSet(df, vocabulary, stack_size, transforms_fn)
    res = generate_dataset(data[0])
    del data[0]
    return res, keyword_num, vocabulary


# @disk_cache(basename="scope_grammar_language_model_load_parsed_data", directory=CACHE_DATA_PATH)
def load_parsed_data(stack_size):
    data = read_monitored_parsed_c99_slk_top_down_code_without_consistent_name()
    for d, n in zip(data, ["train", "val", "test"]):
        print("There are {} raw data in the {} dataset".format(len(d), n))
    res = [0] * 3
    res[0], keyword_num, vocabulary = transform_data_from_df_to_dataset(data, stack_size)
    res[1], keyword_num, vocabulary = transform_data_from_df_to_dataset(data, stack_size)
    res[2], keyword_num, vocabulary = transform_data_from_df_to_dataset(data, stack_size)
    return res, keyword_num, vocabulary


def load_test_data(stack_size, ):
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
    res, keyword_num, vocabulary = transform_data_from_df_to_dataset(data, stack_size)
        # with open(cache_path, 'wb') as handle:
        #     print("dump cache from:{}".format(cache_path))
        #     pickle.dump([res, keyword_num, vocabulary], handle)
    print("parsed test data size:{}".format(len(res)))
    return res, keyword_num, vocabulary


def accuracy_evaluate(model,
                      dataset,
                      batch_size,
                      loss_function,):
    total_loss = torch.Tensor([0]).cuda(GPU_INDEX)
    steps = torch.Tensor([0]).cuda(GPU_INDEX)
    accuracy_dict = None
    for batch_data in data_loader(dataset, batch_size=batch_size, is_shuffle=True, drop_last=True):
        identifier_scope_mask, is_identifier, lengths, target, tokens, update_mask\
            = parse_batch_data(
            batch_data)
        log_probs = model.forward(tokens,
                                  identifier_scope_mask,
                                  is_identifier,
                                  update_mask,
                                  lengths)

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
        stack_size,
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

    stack_size = 10
    data, keyword_num, vocabulary = load_test_data(stack_size)
    only_evaluate(data, keyword_num, vocabulary, 4, 100, 100, 3, 0.01, 50, "scoped_baseline_language_model_1.pkl",
                  stack_size,
                  load_previous_model=True)
    # print(data[0]['code'][0])
    # train_and_evaluate(data, keyword_num, vocabulary, 16, 100, 100, 3, 0.01, 50, "scoped_baseline_language_model_1.pkl",
    #                    stack_size,
    #                    load_previous_model=False)
    # The model c89_grammar_lm_1.pkl best valid perplexity is 2.7838220596313477 and test perplexity is 2.7718544006347656
    # train_and_evaluate(data, keyword_num, vocabulary, 16, 200, 200, 3, 0.01, 50, "scope_grammar_language_model_2.pkl",
    #                    stack_size,
    #                    load_previous_model=False)
    # The model c89_grammar_lm_2.pkl best valid perplexity is 3.062429189682007 and test perplexity is 3.045041799545288
    # train_and_evaluate(data, keyword_num, vocabulary, 16, 300, 300, 3, 0.01, 50, "scope_grammar_language_model_3.pkl",
    #                    stack_size, load_previous_model = False)
    # The model c89_grammar_lm_3.pkl best valid perplexity is 2.888122797012329 and test perplexity is 2.8750290870666504
