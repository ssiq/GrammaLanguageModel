import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.utils.rnn as rnn_util
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms, utils

import pandas as pd

import config
from c_code_processer.code_util import LeafToken, MonitoredParser, parse_tree_to_top_down_process, \
    get_all_c99_production_vocabulary
from common import torch_util
from common import util
from common.util import show_process_map, key_transform, FlatMap
from embedding.wordembedding import Vocabulary, load_vocabulary
from read_data.load_parsed_data import read_parsed_top_down_code, get_token_vocabulary, get_vocabulary_id_map

BEGIN, END, UNK = ["<BEGIN>", "<END>", "<UNK>"]
MAX_LENGTH = 500
GPU_INDEX = 1
PAD_TOKEN = -1

def to_cuda(x):
    return x.cuda(GPU_INDEX)


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
        tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["tokens"]]],
                                                        use_position_label=True)[0]
        sample = {"tree": self.data_df.iloc[index]["parse_tree"],
                  "tokens": tokens[:-1],
                  "target": tokens[1:],
                  }
        return sample

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)


class ProductionSequenceMap(object):
    def __call__(self, sample):
        """
        :param sample: a Parse Node List
        :return: a list of list. The inner list contains a sub process from
        the time of one Terminal token on the top of stack to another time like this
        """
        res = []
        cache = []
        for node in sample:
            if isinstance(node, LeafToken):
                # print("Terminal token:{}".format(node.value))
                res.append(cache)
                cache = []
            else:
                cache.append(node)
        if cache:
            print("There has something left in the cache")
            res.append(cache)
        return res


class ProductionIdMap(object):
    def __init__(self,
                 get_production_id):
        self.get_production_id = get_production_id

    def __call__(self, sample):
        def cal_token_index(seq):
            res = []
            index = 0
            for s in seq:
                index += len(s)
                res.append(index)
                index += 1
            res.append(index)
            return res

        def get_production_id(x):
            res = self.get_production_id(x)
            # print("get production {}'s id is {}".format(x, res))
            return res
        sample = [[get_production_id(token) for token in sub_part] for sub_part in sample]
        predict_index = [cal_token_index(p) for p in sample]
        return {"productions": sample, "predict_index": predict_index}


class SequenceProductionLanguageModel(nn.Module):
    def __init__(self, production_num, token_num, rnn_num_layers, hidden_state_size, embedding_dim, batch_size):
        super().__init__()
        self._production_num = production_num
        self._token_num = token_num
        self._batch_size = batch_size
        self._rnn_num_layers = rnn_num_layers
        self._hidden_state_size = hidden_state_size
        self._embedding_dim = embedding_dim

        self.token_seq_initial_state = self.initial_state()
        self.token_seq_rnn = self.rnn_seq()

        self.production_seq_initial_state = self.initial_state()
        self.production_seq_rnn = self.rnn_seq()

        self.token_embedding = nn.Embedding(token_num, embedding_dim, sparse=True).cpu()
        self.production_embedding = nn.Embedding(production_num, embedding_dim, sparse=True).cpu()

        self.token_to_production_transformation = self.transformation_mlp()
        self.production_to_token_transformation = self.transformation_mlp()

        self.token_predict_mlp = to_cuda(nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, token_num)
        ))

        self.production_predict_mlp = to_cuda(nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, production_num)
        ))

    def initial_state(self):
        return (to_cuda(nn.Parameter(torch.randn((self._rnn_num_layers, self._batch_size, self._hidden_state_size)),
                             requires_grad=True)),
                to_cuda(nn.Parameter(torch.randn((self._rnn_num_layers, self._batch_size, self._hidden_state_size)),
                             requires_grad=True)))

    def rnn_seq(self):
        return to_cuda(nn.LSTM(input_size=self._embedding_dim,
                           hidden_size=self._hidden_state_size,
                           num_layers=self._rnn_num_layers,))

    def transformation_mlp(self):
        return to_cuda(nn.Sequential(
            nn.Linear(self._embedding_dim, self._embedding_dim),
            nn.ReLU(),
            nn.Linear(self._embedding_dim, self._embedding_dim),
        ))

    def _production_embedding(self, production_sequences):
        return [[to_cuda(self.production_embedding(p)) for p in production] for production in production_sequences]

    def _get_batch_from_packed_sequence(self, packed_sequence, batch_sizes):
        res = []
        begin_index = 0
        end_index = 0
        for i in range(batch_sizes):
            end_index += batch_sizes[i]
            now_token_embedding = packed_sequence[begin_index:end_index]
            begin_index += batch_sizes[i]
            res.append(now_token_embedding)
        return res

    def _stack_batch(self, batch):
        res = []
        for i in range(len(batch[0])):
            r = []
            for b in batch:
                if i >= len(b):
                    break
                r.append(b[i])
            res.append(r)
        return res

    def _concat_token_and_production(self, tokens, productions):
        for t in tokens:
            t.unsqueeze_(0)
        res = [tokens[0]]
        for t, p in zip(tokens[1:], productions):
            res.append(p)
            res.append(t)
        return torch.cat(res,)

    def forward(self,
                productions,
                tokens,
                predict_index,
                ):
        packed_sequence, idx_unsort = torch_util.pack_sequence(tokens)
        packed_sequence = rnn_util.PackedSequence(
            to_cuda(self.token_embedding(packed_sequence.data)),
            packed_sequence.batch_sizes,
        )
        token_rnn_features = self.token_seq_rnn(packed_sequence, self.token_seq_initial_state)
        token_rnn_features = rnn_util.PackedSequence(
            self.token_to_production_transformation(token_rnn_features.data),
            token_rnn_features.batch_sizes
        )

        unbinded_token_rnn_features = torch.unbind(token_rnn_features.data)
        token_rnn_features_batch = self._get_batch_from_packed_sequence(unbinded_token_rnn_features,
                                                                        token_rnn_features.batch_sizes)
        token_rnn_features_batch = self._stack_batch(token_rnn_features_batch)
        token_rnn_features_batch = util.index_select(token_rnn_features_batch, idx_unsort)

        production_sequences_embedding = self._production_embedding(productions)
        mixed_production_sequence = []
        for token_feature, production_embedding in zip(token_rnn_features_batch, production_sequences_embedding):
            mixed_production_sequence.append(self._concat_token_and_production(token_feature, production_embedding))

        mixed_production_sequence, unsort_index = torch_util.pack_sequence(mixed_production_sequence, )
        predict_tokens = []
        mixed_production_sequence = self.production_seq_rnn(mixed_production_sequence)


def train(model, train_dataset, batch_size, loss_function, optimizer):
    pass


def evaluate(model, valid_dataset, batch_size, loss_function):
    pass


def train_and_evaluate(data,
                       batch_size,
                       embedding_dim,
                       hidden_state_size,
                       rnn_num_layer,
                       learning_rate,
                       epoches,
                       saved_name,
                       load_previous_model=False):
    save_path = os.path.join(config.save_model_root, saved_name)
    for d, n in zip(data, ["train", "val", "test"]):
        print("There are {} raw data in the {} dataset".format(len(d), n))
    vocabulary = load_vocabulary(get_token_vocabulary, get_vocabulary_id_map, [BEGIN], [END], UNK)
    production_vocabulary = get_all_c99_production_vocabulary()
    print("terminal num:{}".format(len(production_vocabulary._terminal_id_set)))
    transforms_fn = transforms.Compose([
        key_transform(ProductionSequenceMap(), "tree"),
        key_transform(ProductionIdMap(production_vocabulary.get_production_id), "tree"),
        FlatMap(),
    ])
    generate_dataset = lambda df: CCodeDataSet(df, vocabulary, transforms_fn)
    data = [generate_dataset(d) for d in data]
    for d, n in zip(data, ["train", "val", "test"]):
        print("There are {} parsed data in the {} dataset".format(len(d), n))
    train_dataset, valid_dataset, test_dataset = data

    loss_function = nn.CrossEntropyLoss(size_average=False, ignore_index=PAD_TOKEN)
    model = SequenceProductionLanguageModel(

    )
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    if load_previous_model:
        torch_util.load_model(model, save_path)
        valid_loss = evaluate(model, valid_dataset, batch_size, loss_function)
        test_loss = evaluate(model, test_dataset, batch_size, loss_function)
        best_valid_perplexity = torch.exp(valid_loss)[0]
        best_test_perplexity = torch.exp(test_loss)[0]
        print(
            "load the previous mode, validation perplexity is {}, test perplexity is :{}".format(best_valid_perplexity,
                                                                                                 best_test_perplexity))
        scheduler.step(best_valid_perplexity)
    else:
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
    # data = read_parsed_top_down_code(True)
    monitor = MonitoredParser(lex_optimize=False,
                              yacc_debug=True,
                              yacc_optimize=False,
                              yacctab='yacctab')
    code = """
          int add(int a, int b)
          {
              return a+b*c;
          }
          """
    node, _, tokens = monitor.parse_get_production_list_and_token_list(code)
    productions = parse_tree_to_top_down_process(node)
    production_vocabulary = get_all_c99_production_vocabulary()
    transforms_fn = transforms.Compose([
        key_transform(ProductionSequenceMap(), "tree"),
        key_transform(ProductionIdMap(production_vocabulary.get_production_id), "tree"),
        FlatMap(),
    ])
    res = transforms_fn({"tree": productions})['productions']

    res_itr = iter(res)
    t = next(res_itr)
    i = 0
    for n in productions:
        if isinstance(n, LeafToken):
            assert i == len(t)
            t = next(res_itr)
            i = 0
        else:
            print(n)
            print(t[i], production_vocabulary.get_production_by_id(t[i]))
            i += 1

