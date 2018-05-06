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
from c_code_processer.buffered_clex import BufferedCLex
from c_code_processer.code_util import C99ProductionVocabulary, \
    get_all_c99_production_vocabulary, LeafToken, show_production_node, parse_tree_to_top_down_process
from common import torch_util
from common.util import generate_mask, show_process_map, data_loader, padded_to_length, key_transform, FlatMap, IsNone
from embedding.wordembedding import load_vocabulary, Vocabulary
from read_data.load_parsed_data import get_token_vocabulary, get_vocabulary_id_map, read_parsed_top_down_code, \
    read_parsed_slk_top_down_code

from c_code_processer.slk_parser import SLKConstants, LabelVocabulary, C89ProductionVocabulary, slk_parse

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
        # for t in self._samples:
        #     print(t)
        if self.transform:
            self._samples = show_process_map(self.transform, self._samples)
        # for t in self._samples:
        #     print(t)
        # for s in self._samples:
        #     for k, v in s.items():
        #         print("{}:shape {}".format(k, np.array(v).shape))

    def _get_raw_sample(self, index):
        tokens = self.vocabulary.parse_text_without_pad([[k for k in self.data_df.iloc[index]["tokens"]]],
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
                 production_vocabulary: C89ProductionVocabulary):
        self._production_vocabulary = production_vocabulary

    def __call__(self, sample):
        """
        :param sample: a list of node
        :return: a dict of list {'to_parse_token', 'terminal_mask'}
        """
        # print()
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

        terminal_mask = []
        for i, token in enumerate(to_parse_token_id):
            if i in peeked_compact_dict:
                # print("peek", peeked_compact_dict[i])
                terminal_mask.append(generate_mask_fn(peeked_compact_dict[i]))
            else:
                # print("terminal", get_matched_terminal_index(token))
                terminal_mask.append(generate_mask_fn(get_matched_terminal_index(token)))

        return {"to_parse_token": to_parse_token_id, "terminal_mask": terminal_mask,}


class PadMap(object):
    def __init__(self, type_num):
        self._type_num = type_num
        self._mask_pad = [0] * type_num

    def __call__(self, sample: dict):
        def pad_one_sample(x):
            x['tokens'] = padded_to_length(x['tokens'], MAX_LENGTH, 0)
            x['to_parse_token'] = padded_to_length(x['to_parse_token'], MAX_LENGTH, 0)
            x['terminal_mask'] = padded_to_length(x['terminal_mask'], MAX_LENGTH, self._mask_pad)
            x['target'] = padded_to_length(x['target'], MAX_LENGTH, PAD_TOKEN)
            return x
        return pad_one_sample(sample)


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

        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.type_embedding = nn.Embedding(type_num, embedding_dim, sparse=True).cuda(GPU_INDEX)
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
        self._all_type_index = torch.range(0, type_num-1).type(torch.LongTensor).cuda(GPU_INDEX)


    def _embedding(self, token_sequence):
        """
        :param token_sequence: a long variable with the shape [batch, seq]
        :return: The embedding seq of shape [batch, seq, embedding_dim]
        """
        return self.token_embeddings(token_sequence).cuda(GPU_INDEX)

    def initial_state(self):
        return (nn.Parameter(torch.randn((self._rnn_num_layers, self._batch_size, self._hidden_state_size)),
                             requires_grad=True).cuda(GPU_INDEX),
                nn.Parameter(torch.randn((self._rnn_num_layers, self._batch_size, self._hidden_state_size)),
                             requires_grad=True).cuda(GPU_INDEX))

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

        rnn_feature = self._forward_rnn(tokens, length)
        batch_sizes = rnn_feature.batch_sizes
        rnn_feature = rnn_feature.data
        # print("rrn_feature:{}".format(torch.typename(rnn_feature)))
        # print("rnn_feature size:{}".format(rnn_feature.size()))

        # to_parse_token.register_hook(create_hook_fn("to_parse_token"))
        to_parse_token = torch.nn.utils.rnn.pack_padded_sequence(to_parse_token, length, batch_first=True).data
        # print("to_parse_token1:{}".format(torch.typename(to_parse_token)))
        to_parse_token = self.type_embedding(to_parse_token.cuda(GPU_INDEX))
        # print("to_parse_token2:{}".format(torch.typename(to_parse_token)))
        # print("to_parse_token embedding size:{}".format(to_parse_token.size()))

        ternimal_token_probability = self._output_forward(rnn_feature, to_parse_token)
        # print("ternimal_token_probability:{}".format(torch.typename(ternimal_token_probability)))
        # ternimal_token_probability.register_hook(create_hook_fn("ternimal_token_probability before mask softmax"))
        # print("terminal_token_probability size:{}".format(ternimal_token_probability.size()))
        terminal_mask = torch.nn.utils.rnn.pack_padded_sequence(terminal_mask.type(torch.FloatTensor).cuda(GPU_INDEX),
                                                                length, batch_first=True).data
        # print("terminal_mask:{}".format(torch.typename(terminal_mask)))
        # print("terminal mask size:{}".format(terminal_mask.size()))
        ternimal_token_probability = torch_util.mask_softmax(ternimal_token_probability,
                                                             terminal_mask.cuda(GPU_INDEX))
        # print("ternimal_token_probability:{}".format(torch.typename(ternimal_token_probability)))
        # ternimal_token_probability.register_hook(create_hook_fn("ternimal_token_probability"))
        # print("masked terminal_token_probability size:{}".format(ternimal_token_probability.size()))

        type_feature_predict = self.type_feature_mlp(self.type_embedding(self._all_type_index))
        # print("type_feature_predict:{}".format(torch.typename(type_feature_predict)))
        # type_feature_predict.register_hook(create_hook_fn("type_feature_predict"))
        # print("type_feature_predict size:{}".format(type_feature_predict.size()))
        rnn_feature_predict = self.rnn_feature_mlp(rnn_feature)
        # print("rnn_feature_predict:{}".format(torch.typename(rnn_feature_predict)))
        # rnn_feature_predict.register_hook(create_hook_fn("rnn_feature_predict"))
        # print("rnn_feature_predict size:{}".format(rnn_feature_predict.size()))
        # ternimal_token_probability = autograd.Variable(torch_util.to_sparse(ternimal_token_probability,
        #                                                                     gpu_index=GPU_INDEX))
        predict = F.log_softmax(rnn_feature_predict+torch.mm(ternimal_token_probability, type_feature_predict), dim=-1)
        # print("predict:{}".format(torch.typename(predict)))
        # print("predict size:{}".format(predict.size()))
        # predict.register_hook(create_hook_fn("predict"))
        predict_log = torch.nn.utils.rnn.PackedSequence(predict, batch_sizes)
        # predict_log.register_hook(create_hook_fn("predict_log1"))
        predict_log, _ = torch.nn.utils.rnn.pad_packed_sequence(predict_log, batch_first=True, padding_value=PAD_TOKEN)
        # predict_log.register_hook(create_hook_fn("predict_log"))
        unpacked_out = torch.index_select(predict_log, 0, autograd.Variable(idx_unsort).cuda(GPU_INDEX))
        # print("unpacked_out:{}".format(torch.typename(unpacked_out)))
        # unpacked_out.register_hook(create_hook_fn("unpacked_out"))
        return unpacked_out

def to_numpy(var):
    namelist = torch.typename(var).split('.')
    if "sparse" in namelist:
        var = var.to_dense()
    return var.cpu().numpy()

HAS_NAN = False
def is_nan(var):
    if var is None:
        return "None"
    res = np.isnan(np.sum(to_numpy(var)))
    if res:
        global HAS_NAN
        HAS_NAN = True
    return res

def show_tensor(var):
    if var is None:
        return "None"
    var = to_numpy(var)
    return "all zeros:{}, has nan:{}, value:{}".format(np.all(var==0), np.isnan(np.sum(var)), var)

def create_hook_fn(name):
    def p(v):
        print("{} gradient: is nan {}".format(name, is_nan(v.detach())))
    return p


def train(model,
          dataset,
          batch_size,
          loss_function,
          optimizer):
    total_loss = torch.Tensor([0])
    steps = torch.Tensor([0])
    for batch_data in data_loader(dataset, batch_size=batch_size, is_shuffle=True,  drop_last=True, epoch_ratio=0.5):
        # print(batch_data['terminal_mask'])
        # print('batch_data size: ', len(batch_data['terminal_mask'][0]), len(batch_data['terminal_mask'][0][0]))
        # res = list(more_itertools.collapse(batch_data['terminal_mask']))
        # print('res len: ', len(res))
        # res = util.padded(batch_data['terminal_mask'], deepcopy=True, fill_value=0)
        # print('batch_data size: ', len(res[0]), len(res[0][0]))
        # res = list(more_itertools.collapse(res))
        # print('res len: ', len(res))
        target = batch_data["target"]
        del batch_data["target"]
        model.zero_grad()
        batch_data = {k: autograd.Variable(torch.LongTensor(v)) for k, v in batch_data.items()}
        log_probs = model.forward(**batch_data)
        # log_probs.register_hook(create_hook_fn("log_probs"))

        batch_log_probs = log_probs.view(-1, list(log_probs.size())[-1])

        target, idx_unsort = torch_util.pack_padded_sequence(
            autograd.Variable(torch.LongTensor(target)).cuda(GPU_INDEX),
            batch_data['length'], batch_firse=True, GPU_INDEX=GPU_INDEX)
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
        target = batch_data["target"]
        del batch_data["target"]
        batch_data = {k: autograd.Variable(torch.LongTensor(v)) for k, v in batch_data.items()}
        log_probs = model.forward(**batch_data)

        batch_log_probs = log_probs.view(-1, list(log_probs.size())[-1])
        target, idx_unsort = torch_util.pack_padded_sequence(autograd.Variable(torch.LongTensor(target)).cuda(GPU_INDEX),
                                                             batch_data['length'], batch_firse=True, GPU_INDEX=GPU_INDEX)
        target, _ = torch_util.pad_packed_sequence(target, idx_unsort, pad_value=PAD_TOKEN, batch_firse=True,
                                                GPU_INDEX=GPU_INDEX)

        loss = loss_function(batch_log_probs, target.view(-1))
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
                       saved_name,
                       load_previous_model=False):
    save_path = os.path.join(config.save_model_root, saved_name)
    for d, n in zip(data, ["train", "val", "test"]):
        print("There are {} raw data in the {} dataset".format(len(d), n))
    vocabulary = load_vocabulary(get_token_vocabulary, get_vocabulary_id_map, [BEGIN], [END], UNK)
    slk_constants = SLKConstants()
    label_vocabulary = LabelVocabulary(slk_constants)
    production_vocabulary = C89ProductionVocabulary(slk_constants)
    transforms_fn = transforms.Compose([
        IsNone("original"),
        key_transform(GrammarLanguageModelTypeInputMap(production_vocabulary), "tree"),
        IsNone("after type input"),
        FlatMap(),
        IsNone("Flat Map"),
        PadMap(production_vocabulary.token_num()),
        IsNone("Pad Map"),
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
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    data = read_parsed_slk_top_down_code(False)
    train_and_evaluate(data, 16, 100, 100, 3, 0.001, 30, "c89_grammar_lm_1.pkl", load_previous_model=False)
    #The model c89_grammar_lm_1.pkl best valid perplexity is 2.7838220596313477 and test perplexity is 2.7718544006347656
    train_and_evaluate(data, 16, 200, 200, 3, 0.001, 30, "c89_grammar_lm_2.pkl", load_previous_model=True)
    #The model c89_grammar_lm_2.pkl best valid perplexity is 3.062429189682007 and test perplexity is 3.045041799545288
    train_and_evaluate(data, 16, 300, 300, 3, 0.001, 40, "c89_grammar_lm_3.pkl", load_previous_model=True)
    #The model c89_grammar_lm_3.pkl best valid perplexity is 2.888122797012329 and test perplexity is 2.8750290870666504
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
    # # node, _, tokens = monitor.parse_get_production_list_and_token_list(code)
    # clex = BufferedCLex(error_func=lambda self, msg, line, column: None,
    #                     on_lbrace_func=lambda: None,
    #                     on_rbrace_func=lambda: None,
    #                     type_lookup_func=lambda typ: None)
    # clex.build()
    # node, tokens = slk_parse(code, clex)
    # for token in tokens:
    #     print(token)
    # # show_production_node(node)
    #
    # # productions = parse_tree_to_top_down_process(node)
    # for p in node:
    #     print(p)
    # # production_vocabulary = get_all_c99_production_vocabulary()
    # slk_constants = SLKConstants()
    # production_vocabulary = C89ProductionVocabulary(slk_constants)
    # input_map = GrammarLanguageModelTypeInputMap(production_vocabulary)
    # input_f = input_map(node)
    # print("length of token:{}".format(len(tokens)))
    # print("length of to parse token:{}".format(len(input_f['to_parse_token'])))
