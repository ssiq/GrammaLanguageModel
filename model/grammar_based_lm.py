import more_itertools
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torch import multiprocessing

import numpy as np
import pandas as pd

import typing
import os

import config
from c_code_processer.code_util import parse_tree_to_top_down_process, ProductionVocabulary, \
    get_all_c99_production_vocabulary, LeafToken, MonitoredParser, show_production_node
from common import util, torch_util
from common.util import generate_mask, show_process_map, data_loader
from embedding.wordembedding import load_vocabulary, Vocabulary
from read_data.load_parsed_data import get_token_vocabulary, get_vocabulary_id_map, read_parsed_tree_code, \
    read_parsed_top_down_code

BEGIN, END, UNK = ["<BEGIN>", "<END>", "<UNK>"]
PAD_TOKEN = -1
GPU_INDEX = 1


class CCodeDataSet(Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary: Vocabulary,
                 transform=None):
        self.data_df = data_df[data_df['tokens'].map(lambda x: x is not None)]
        self.data_df = self.data_df[self.data_df['tokens'].map(lambda x: len(x) < 500)]
        self.transform = transform
        self.vocabulary = vocabulary

        self._samples = [self._get_raw_sample(i) for i in range(len(self.data_df))]
        if self.transform:
            self._samples = show_process_map(self.transform, self._samples)
        # for s in self._samples:
        #     for k, v in s.items():
        #         print("{}:shape {}".format(k, np.array(v).shape))

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

        stack = [get_token_id(production_vocabulary.EMPTY), sample[0].left_id]
        to_parse_token_id = [sample[0].left_id]

        for node in sample:
            type_id = stack.pop()
            if isinstance(node, LeafToken):
                # print("Terminal token:{}".format(node.value))
                to_parse_token_id.append(stack[-1])
            else:
                assert type_id == node.left_id
                for right_id in reversed(get_node_right_id(node)):
                    stack.append(right_id)

        terminal_mask = [generate_mask_fn(get_matched_terminal_index(token)) for token in to_parse_token_id]
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


def key_transform(transform, key, ):
    def transform_fn(sample):
        sample[key] = transform(sample[key])
        return sample

    return transform_fn


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

        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True).cpu()
        self.type_embedding = nn.Embedding(type_num, embedding_dim, sparse=True).cpu()
        self.rnn = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_state_size,
                           num_layers=rnn_num_layers,).cuda(GPU_INDEX)

        self.token_prob_mlp = nn.Sequential(
            nn.Linear(hidden_state_size+embedding_dim, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, type_num)).cuda(GPU_INDEX)

        self.type_feature_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, vocab_size)
        ).cuda(GPU_INDEX)

        self.rnn_feature_mlp = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, vocab_size),
        ).cuda(GPU_INDEX)

        self._initial_state = self.initial_state()
        self._all_type_index = torch.range(0, type_num-1).type(torch.LongTensor)


    def _embedding(self, token_sequence):
        """
        :param token_sequence: a long variable with the shape [batch, seq]
        :return: The embedding seq of shape [batch, seq, embedding_dim]
        """
        return self.token_embeddings(token_sequence).cuda(GPU_INDEX)

    def initial_state(self):
        return (autograd.Variable(torch.randn((self._rnn_num_layers, self._batch_size, self._hidden_state_size)),
                                  requires_grad=True).cuda(GPU_INDEX),
                autograd.Variable(torch.randn((self._rnn_num_layers, self._batch_size, self._hidden_state_size)),
                                  requires_grad=True).cuda(GPU_INDEX))

    def _forward_rnn(self,
                     embedding_sequence,
                     lengths):
        """
        :param embedding_sequence: a float variable with the shape [batch, seq, embedding_dim]
        :param lengths: a long variable with the shape [batch, ]
        :return: a float variable with the shape [batch, seq, feature]
        """
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(embedding_sequence, lengths, batch_first=True)
        # print("packed seq size:{}".format(packed_seq.data.data.size()))
        output, _ = self.rnn(packed_seq, self._initial_state)
        return output

    def _output_forward(self, rnn_features, to_parse_token):
        in_features = torch.cat([rnn_features, to_parse_token], dim=-1)
        # print("in_feature size:{}".format(in_features.size()))
        return self.token_prob_mlp(in_features)

    def forward(self,
                tokens,
                to_parse_token,
                terminal_mask,
                length):
        """
        :param tokens: a long variable with the shape [batch, seq]
        :param to_parse_token: a long variable with the shape [batch, seq]
        :param terminal_mask: a long variable with the shape [batch, seq, token_type_number]
        :param length: a long variable with the shape [batch, ]
        :return: a float variable with the shape [batch, seq, vovabulary_size]
        """
        _, idx_sort = torch.sort(length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        tokens, to_parse_token,terminal_mask,  length = \
            [torch.index_select(t, 0, idx_sort) for t in [tokens, to_parse_token,terminal_mask,length]]
        length = list(length)

        embedding_feature = self._embedding(tokens)
        rnn_feature = self._forward_rnn(embedding_feature, length)
        batch_sizes = rnn_feature.batch_sizes
        rnn_feature = rnn_feature.data
        # print("rnn_feature size:{}".format(rnn_feature.size()))

        to_parse_token = self.type_embedding(to_parse_token).cuda(GPU_INDEX)
        to_parse_token = torch.nn.utils.rnn.pack_padded_sequence(to_parse_token, length, batch_first=True).data
        # print("to_parse_token embedding size:{}".format(to_parse_token.size()))

        ternimal_token_probability = self._output_forward(rnn_feature, to_parse_token)
        # print("terminal_token_probability size:{}".format(ternimal_token_probability.size()))
        terminal_mask = torch.nn.utils.rnn.pack_padded_sequence(terminal_mask, length, batch_first=True).data
        # print("terminal mask size:{}".format(terminal_mask.size()))
        ternimal_token_probability = torch_util.mask_softmax(ternimal_token_probability,
                                                             terminal_mask.type(torch.FloatTensor).cuda(GPU_INDEX))
        # print("masked terminal_token_probability size:{}".format(ternimal_token_probability.size()))

        type_feature_predict = self.type_feature_mlp(self.type_embedding(self._all_type_index).cuda(GPU_INDEX))
        # print("type_feature_predict size:{}".format(type_feature_predict.size()))
        rnn_feature_predict = self.rnn_feature_mlp(rnn_feature)
        # print("rnn_feature_predict size:{}".format(rnn_feature_predict.size()))
        ternimal_token_probability = autograd.Variable(torch_util.to_sparse(ternimal_token_probability,
                                                                            gpu_index=GPU_INDEX))
        predict = F.softmax(rnn_feature_predict+torch.mm(ternimal_token_probability, type_feature_predict), dim=-1)
        # print("predict size:{}".format(predict.size()))
        predict_log = torch.nn.utils.rnn.PackedSequence(torch.log(predict), batch_sizes)
        predict_log, _ = torch.nn.utils.rnn.pad_packed_sequence(predict_log, batch_first=True, padding_value=PAD_TOKEN)
        unpacked_out = torch.index_select(predict_log, 0, autograd.Variable(idx_unsort).cuda(GPU_INDEX))
        return unpacked_out


def train(model,
          dataset,
          batch_size,
          loss_function,
          optimizer):
    total_loss = torch.Tensor([0])
    steps = torch.Tensor([0])
    for batch_data in data_loader(dataset, batch_size=batch_size, is_shuffle=True,  drop_last=True):
        # print(batch_data['terminal_mask'])
        # print('batch_data size: ', len(batch_data['terminal_mask'][0]), len(batch_data['terminal_mask'][0][0]))
        # res = list(more_itertools.collapse(batch_data['terminal_mask']))
        # print('res len: ', len(res))
        # res = util.padded(batch_data['terminal_mask'], deepcopy=True, fill_value=0)
        # print('batch_data size: ', len(res[0]), len(res[0][0]))
        # res = list(more_itertools.collapse(res))
        # print('res len: ', len(res))
        batch_data = {k: util.padded(v, deepcopy=True, fill_value=0 if k!="target" else PAD_TOKEN) for k, v in batch_data.items()}
        target = batch_data["target"]
        del batch_data["target"]
        model.zero_grad()
        batch_data = {k: autograd.Variable(torch.LongTensor(v)) for k, v in batch_data.items()}
        log_probs = model.forward(**batch_data)

        batch_log_probs = log_probs.view(-1, list(log_probs.size())[-1])
        target = list(more_itertools.flatten(target))

        loss = loss_function(batch_log_probs, autograd.Variable(torch.LongTensor(target)).cuda(GPU_INDEX))

        loss.backward()
        optimizer.step()

        total_loss += loss.data.cpu()
        steps += torch.sum(batch_data['length'].data.cpu())
    return total_loss/steps


def evaluate(model,
             dataset,
             batch_size,
             loss_function):
    total_loss = torch.Tensor([0])
    steps = torch.Tensor([0])
    for batch_data in data_loader(dataset, batch_size=batch_size, is_shuffle=True, drop_last=True):
        batch_data = {k: util.padded(v, deepcopy=True, fill_value=0 if k != "target" else -1) for k, v in
                      batch_data.items()}
        target = batch_data["target"]
        del batch_data["target"]
        model.zero_grad()
        batch_data = {k: autograd.Variable(torch.LongTensor(v)) for k, v in batch_data.items()}
        log_probs = model.forward(**batch_data)

        batch_log_probs = log_probs.view(-1, list(log_probs.size())[-1])
        target = list(more_itertools.flatten(target))

        loss = loss_function(batch_log_probs, autograd.Variable(torch.LongTensor(target)).cuda(GPU_INDEX))
        total_loss += loss.data.cpu()
        steps += torch.sum(batch_data['length'].data.cpu())
    return total_loss / steps



def train_and_evaluate(data,
                       batch_size,
                       embedding_dim,
                       hidden_state_size,
                       rnn_num_layer,
                       learning_rate,
                       epoches,
                       saved_name):
    save_path = os.path.join(config.save_model_root, saved_name)
    for d, n in zip(data, ["train", "val", "test"]):
        print("There are {} raw data in the {} dataset".format(len(d), n))
    vocabulary = load_vocabulary(get_token_vocabulary, get_vocabulary_id_map, [BEGIN], [END], UNK)
    production_vocabulary = get_all_c99_production_vocabulary()
    print("terminal num:{}".format(len(production_vocabulary._terminal_id_set)))
    transforms_fn = transforms.Compose([
        key_transform(GrammarLanguageModelTypeInputMap(production_vocabulary), "tree"),
        FlatMap(),
    ])
    generate_dataset = lambda df: CCodeDataSet(df, vocabulary, transforms_fn)
    data = [generate_dataset(d) for d in data]
    for d, n in zip(data, ["train", "val", "test"]):
        print("There are {} parsed data in the {} dataset".format(len(d), n))
    train_dataset, valid_dataset, test_dataset = data

    loss_function = nn.CrossEntropyLoss(size_average=False, ignore_index=PAD_TOKEN)
    model = GrammarLanguageModel(
        vocabulary.vocabulary_size,
        production_vocabulary.token_num(),
        embedding_dim,
        hidden_state_size,
        rnn_num_layer,
        batch_size
    )
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    best_valid_perplexity = None
    best_test_perplexity = None
    for epoch in range(epoches):
        train_loss = train(model, train_dataset, batch_size, loss_function, optimizer)
        valid_loss = evaluate(model, valid_dataset, batch_size, loss_function)
        test_loss = evaluate(model, test_dataset, batch_size, loss_function)

        train_perplexity = torch.exp(train_loss)[0]
        valid_perplexity = torch.exp(valid_loss)[0]
        test_perplexity = torch.exp(test_loss)[0]

        scheduler.step(valid_perplexity)

        if best_valid_perplexity is None or valid_perplexity < best_valid_perplexity:
            best_valid_perplexity = valid_perplexity
            best_test_perplexity = test_perplexity
            torch_util.save_model(model, save_path)

        print("epoch {}: train perplexity of {},  valid perplexity of {}, test perplexity of {}".
              format(epoch, train_perplexity, valid_perplexity, test_perplexity))
    print("The model {} best valid perplexity is {} and test perplexity is {}".
          format(saved_name, best_valid_perplexity, best_test_perplexity))


if __name__ == '__main__':
    data = read_parsed_top_down_code(debug=True)
    train_and_evaluate(data, 2, 100, 100, 3, 0.001, 10, "grammar_lm_test.pkl")
    # monitor = MonitoredParser(lex_optimize=False,
    #                           yacc_debug=True,
    #                           yacc_optimize=False,
    #                           yacctab='yacctab')
    # code = """
    #         int add(int a, int b)
    #         {
    #             return a+b*c;
    #         }
    #         """
    # node, _, tokens = monitor.parse_get_production_list_and_token_list(code)
    # for token in tokens:
    #     print(token)
    # show_production_node(node)
    # productions = parse_tree_to_top_down_process(node)
    # for p in productions:
    #     print(p)
    # production_vocabulary = get_all_c99_production_vocabulary()
    # input_map = GrammarLanguageModelTypeInputMap(production_vocabulary)
    # input_f = input_map(productions)
    # print("length of token:{}".format(len(tokens)))
    # print("length of to parse token:{}".format(len(input_f['to_parse_token'])))
