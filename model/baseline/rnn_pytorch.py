import torch
import torch.nn as nn
import torch.autograd as autograd
import config
import os
import more_itertools

from read_data.load_parsed_data import read_filtered_without_include_code_tokens, get_token_vocabulary, \
    get_vocabulary_id_map
from embedding.wordembedding import load_vocabulary
from common.util import batch_holder
from common import util, torch_util
from sklearn.utils import shuffle
import sys


gpu_index = 0
BEGIN, END, UNK = ["<BEGIN>", "<END>", "<UNK>"]


class LSTMModel(nn.Module):

    def __init__(self, dictionary_size, embedding_dim, hidden_size, num_layers, batch_size, bidirectional=False, dropout=0):
        super(LSTMModel, self).__init__()
        self.dictionary_size = dictionary_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.drop = nn.Dropout(dropout).cuda(gpu_index)

        print('dictionary_size: {}, embedding_dim: {}, hidden_size: {}, num_layers: {}, batch_size: {}, bidirectional: {}, dropout: {}'.format(
            dictionary_size, embedding_dim, hidden_size, num_layers, batch_size, bidirectional, dropout))

        self.bidirectional_num = 2 if bidirectional else 1


        print('before create embedding')
        self.word_embeddings = nn.Embedding(num_embeddings=dictionary_size, embedding_dim=embedding_dim, padding_idx=0).cuda(gpu_index)
        print('before create lstm')
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, dropout=dropout).cuda(gpu_index)
        print('before create tag')
        self.hidden2tag = nn.Linear(hidden_size * self.bidirectional_num, dictionary_size).cuda(gpu_index)

        print('before init hidden')
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(self.num_layers * self.bidirectional_num, self.batch_size, self.hidden_size)).cuda(gpu_index),
                autograd.Variable(torch.randn(self.num_layers * self.bidirectional_num, self.batch_size, self.hidden_size)).cuda(gpu_index))

    def forward(self, inputs, token_lengths):
        """
        inputs: [batch_size, code_length]
        token_lengths: [batch_size, ]
        :param inputs:
        :return:
        """
        # hidden = self.init_hidden()
        inputs = torch.LongTensor(inputs)
        token_lengths = torch.LongTensor(token_lengths)

        _, idx_sort = torch.sort(token_lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        inputs = torch.index_select(inputs, 0, idx_sort)
        token_lengths = list(torch.index_select(token_lengths, 0, idx_sort))

        print('input_size: ', inputs.size())

        embeds = self.word_embeddings(autograd.Variable(inputs).cuda(gpu_index)).view(self.batch_size, -1, self.embedding_dim).cuda(gpu_index)
        # print('embeds_size: {}, embeds is cuda: {}'.format(embeds.size(), embeds.is_cuda))
        embeds = embeds.view(self.batch_size, -1, self.embedding_dim)
        embeds = self.drop(embeds).cuda(gpu_index)
        print('embeds_size: {}, embeds is cuda: {}'.format(embeds.size(), embeds.is_cuda))
        # print('embeds value: {}'.format(embeds.data))
        # print('after embeds token_length: {}'.format(token_lengths))
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(embeds, token_lengths, batch_first=True)
        # print('packed_inputs batch size: ', len(packed_inputs.batch_sizes))
        # print('packed_inputs is cuda: {}'.format(packed_inputs.data.is_cuda))
        lstm_out, self.hidden = self.lstm(packed_inputs, self.hidden)

        unpacked_lstm_out, unpacked_lstm_length = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True,
                                                                               padding_value=0)
        unpacked_lstm_out = self.drop(unpacked_lstm_out)
        dict_output = self.hidden2tag(unpacked_lstm_out).cuda(gpu_index)
        packed_dict_output = torch.nn.utils.rnn.pack_padded_sequence(dict_output, token_lengths, batch_first=True)


        # print('lstm_out batch size: ', len(lstm_out.batch_sizes))
        # print('lstm_out is cuda: ', lstm_out.data.is_cuda)
        # print('lstm value: {}'.format(lstm_out.data))

        # packed_output = nn.utils.rnn.PackedSequence(self.hidden2tag(lstm_out.data).cuda(gpu_index), lstm_out.batch_sizes)    # output shape: [batch_size, token_length, dictionary_size]
        # print('packed_output batch size: ', len(packed_output.batch_sizes))
        # print('packed_output is cuda: ', packed_output.data.is_cuda)

        unpacked_out, unpacked_length = torch.nn.utils.rnn.pad_packed_sequence(packed_dict_output, batch_first=True, padding_value=0)
        # print('unpacked_out: {}, unpacked_length: {}'.format(unpacked_out.size(), unpacked_length))
        unpacked_out = torch.index_select(unpacked_out, 0, autograd.Variable(idx_unsort).cuda(gpu_index))
        # print('unsort unpacked_out: {}'.format(unpacked_out.size()))
        # print('unsort unpacked_out is cuda: {}'.format(unpacked_out.is_cuda))

        return unpacked_out


def train(model, X, y, optimizer, loss_function, batch_size):
    # print('in train')
    steps = 0
    total_loss = torch.Tensor([0])
    print('before shuffle')
    X, y = shuffle(X, y, n_samples=int(len(X)/4))
    print('finish shuffle: x: {}, y: {}'.format(len(X), len(y)))
    batch_token_count = 0
    model.train()

    for inp, out in batch_holder(X, y, batch_size=batch_size)():
        if len(inp) != batch_size:
            break

        # print('in one batch: X: {},{}, y: {},{}'.format(len(inp), len(inp[0]), len(out), len(out[0])))
        # print('X size: ', torch.Tensor(inp).size())
        # print('y size: ', torch.Tensor(y).size())
        inp, inp_len = list(zip(*inp))
        one_batch_count = 0
        for le in inp_len:
            one_batch_count += le
        batch_token_count += one_batch_count
        # print(type(inp), type(inp[0]))
        inp = util.padded(list(inp), deepcopy=True, fill_value=0)
        out = util.padded(list(out), deepcopy=True, fill_value=0)
        # print('input: {}'.format(inp))
        # print('output: {}'.format(out))
        # print('inp[0]: {}, inp[1]: {}, inp[2]: {}, inp[3]: {}'.format(len(inp[0]), len(inp[1]), len(inp[2]), len(inp[3])))
        # print('in one batch: X: {},{}, y: {},{}'.format(len(inp), len(inp[0]), len(out), len(out[0])))
        # print('X size: ', torch.Tensor(inp).size())
        # print('y size: ', torch.Tensor(out).size())

        model.zero_grad()

        model.hidden = model.init_hidden()

        log_probs = model.forward(inp, inp_len)
        # print('log_probs: {}'.format(log_probs.data))

        batch_log_probs = log_probs.view(-1, list(log_probs.size())[-1])
        out = list(more_itertools.flatten(out))

        if steps % 100 == 0:
            _, max_res = torch.max(batch_log_probs, dim=1)
            # print('max_res: {}'.format(list(max_res.data)))
            # print('target: {}'.format(out))
            max_bi = list(zip(max_res.tolist(), out))
            print('max_bi: {}'.format(max_bi))

        loss = loss_function(batch_log_probs, autograd.Variable(torch.LongTensor(out)).cuda(gpu_index))
        print('step {} loss: {}'.format(steps, loss.data))
        total_loss += (loss.data.cpu() * one_batch_count)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        if steps % 1000 == 0:
            sys.stdout.flush()
            sys.stderr.flush()

        steps += 1
    print('mean loss per steps: ', total_loss/steps)
    print('mean loss per token: ', total_loss/batch_token_count)
    return total_loss / batch_token_count


def evaluate(model, X, y, loss_function, batch_size):
    # print('in evaluate')
    steps = 0
    total_loss = torch.Tensor([0])
    batch_token_count = 0
    model.eval()

    for inp, out in batch_holder(X, y, batch_size=batch_size)():
        if len(inp) != batch_size:
            break

        # print('in one batch: X: {},{}, y: {},{}'.format(len(inp), len(inp[0]), len(out), len(out[0])))
        # print('X size: ', torch.Tensor(inp).size())
        # print('y size: ', torch.Tensor(y).size())
        inp, inp_len = list(zip(*inp))
        one_batch_count = 0
        for le in inp_len:
            one_batch_count += le
        batch_token_count += one_batch_count
        # print(type(inp), type(inp[0]))
        inp = util.padded(list(inp), deepcopy=True, fill_value=0)
        out = util.padded(list(out), deepcopy=True, fill_value=0)
        # print('inp[0]: {}, inp[1]: {}, inp[2]: {}, inp[3]: {}'.format(len(inp[0]), len(inp[1]), len(inp[2]), len(inp[3])))
        # print('in one batch: X: {},{}, y: {},{}'.format(len(inp), len(inp[0]), len(out), len(out[0])))
        # print('X size: ', torch.Tensor(inp).size())
        # print('y size: ', torch.Tensor(out).size())
        # print('token_length size: {}'.format(inp_len))

        model.hidden = model.init_hidden()

        log_probs = model.forward(inp, inp_len)

        batch_log_probs = log_probs.view(-1, list(log_probs.size())[-1])
        out = list(more_itertools.flatten(out))

        loss = loss_function(batch_log_probs, autograd.Variable(torch.LongTensor(out)).cuda(gpu_index))
        print('step {} loss: {}'.format(steps, loss.data))
        total_loss += (loss.data.cpu() * one_batch_count)

        if steps % 1000 == 0:
            sys.stdout.flush()
            sys.stderr.flush()

        steps += 1
    print('mean loss per steps: ', total_loss / steps)
    print('mean loss per token: ', total_loss / batch_token_count)
    return total_loss / batch_token_count


def parse_xy(codes, word_to_id_fn):
    begin_tokens = [BEGIN]
    end_tokens = [END]

    codes = [begin_tokens + code + end_tokens for code in codes]
    codes = [[word_to_id_fn(token) + 1 for token in code] for code in codes]
    codes = list(filter(lambda x: len(x) < 500, codes))

    X = []
    y = []
    for code in codes:
        inp = code[:-1]
        out = code[1:]
        X += [(inp, len(inp))]
        y += [out]
    return X, y


def train_and_evaluate_lstm_model(embedding_dim, hidden_size, num_layers, bidirectional, dropout, learning_rate, batch_size, epoches, saved_name, load_path=None):
    print('------------------------------------- start train and evaluate ----------------------------------------')
    print('embedding_dim: {}, hidden_size: {}, num_layers: {}, bidirectional: {}, dropout: {}, '
          'learning_rate: {}, batch_size: {}, epoches: {}, saved_name: {}'.format(
        embedding_dim, hidden_size, num_layers, bidirectional, dropout, learning_rate, batch_size, epoches, saved_name))
    debug = False
    save_path = os.path.join(config.save_model_root, saved_name)

    vocabulary = load_vocabulary(get_token_vocabulary,
                                 get_vocabulary_id_map,
                                 [BEGIN],
                                 [END],
                                 UNK)
    vocabulary_size = vocabulary.vocabulary_size

    print('before read data')
    if debug:
        train_data, valid_data, test_data = [d[:100] for d in read_filtered_without_include_code_tokens()]
    else:
        train_data, valid_data, test_data = read_filtered_without_include_code_tokens()

    print("train data size:{}".format(len(train_data)))

    print('before create loss function')
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    print('before create model')
    model = LSTMModel(vocabulary_size + 1, embedding_dim, hidden_size, num_layers, batch_size, bidirectional, dropout)
    if load_path is not None:
        load_path = os.path.join(config.save_model_root, load_path)
        print('load model from {}'.format(load_path))
        torch_util.load_model(model, load_path)
    print('after create model')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    print('after create optimizer')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


    print('before parse xy')
    train_X, train_y = parse_xy(train_data, vocabulary.word_to_id)
    valid_X, valid_y = parse_xy(valid_data, vocabulary.word_to_id)
    test_X, test_y = parse_xy(test_data, vocabulary.word_to_id)
    print('after parse xy train data: {}, valid data: {}, test data: {}'.format(len(train_X), len(valid_X), len(test_X)))

    best_valid_perplexity = None
    best_test_perplexity = None

    sys.stdout.flush()
    sys.stderr.flush()

    for i in range(epoches):
        print('in epoch {}'.format(i))
        train_loss = train(model, train_X, train_y, optimizer, loss_function, batch_size)
        print('after train: {}'.format(train_loss))
        valid_loss = evaluate(model, valid_X, valid_y, loss_function, batch_size)
        print('after valid: {}'.format(valid_loss))
        test_loss = evaluate(model, test_X, test_y, loss_function, batch_size)
        print('after test: {}'.format(test_loss))
        print("epoch {}: train loss of {},  valid loss of {}, test loss of {}".
              format(i, train_loss, valid_loss, test_loss))

        train_perplexity = torch.exp(train_loss)[0]
        valid_perplexity = torch.exp(valid_loss)[0]
        test_perplexity = torch.exp(test_loss)[0]

        scheduler.step(valid_perplexity)

        if best_valid_perplexity is None or valid_perplexity < best_valid_perplexity:
            best_valid_perplexity = valid_perplexity
            best_test_perplexity = test_perplexity
            torch_util.save_model(model, save_path)

        print("epoch {}: train perplexity of {},  valid perplexity of {}, test perplexity of {}".
              format(i, train_perplexity, valid_perplexity, test_perplexity))
        sys.stdout.flush()
        sys.stderr.flush()
    print("The model {} best valid perplexity is {} and test perplexity is {}".
          format(saved_name, best_valid_perplexity, best_test_perplexity))


def read_wiki_dataset():
    from torchtext import datasets
    train_iter, valid_iter, test_iter = datasets.WikiText2.iters(batch_size=4, bptt_len=30)



if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    print('start')

    train_and_evaluate_lstm_model(embedding_dim=300, hidden_size=200, num_layers=2, bidirectional=False, dropout=0.35, learning_rate=0.005, batch_size=16, epoches=40, saved_name='neural_lstm_1.pkl', load_path='neural_lstm_1.pkl')
    train_and_evaluate_lstm_model(embedding_dim=100, hidden_size=100, num_layers=1, bidirectional=False, dropout=0.35, learning_rate=0.005, batch_size=16, epoches=40, saved_name='neural_lstm_2.pkl', load_path='neural_lstm_2.pkl')
    train_and_evaluate_lstm_model(embedding_dim=300, hidden_size=200, num_layers=1, bidirectional=False, dropout=0.35, learning_rate=0.005, batch_size=16, epoches=40, saved_name='neural_lstm_3.pkl', load_path='neural_lstm_3.pkl')
    train_and_evaluate_lstm_model(embedding_dim=100, hidden_size=100, num_layers=2, bidirectional=False, dropout=0.35, learning_rate=0.005, batch_size=16, epoches=40, saved_name='neural_lstm_4.pkl', load_path='neural_lstm_4.pkl')

    # train_and_evaluate_lstm_model(embedding_dim=100, hidden_size=100, num_layers=3, bidirectional=False, dropout=0.35, learning_rate=0.005, batch_size=16, epoches=10, saved_name='neural_lstm_5.pkl', load_path='neural_lstm_5.pkl')
        # final train perplexity of 9.800065994262695,  valid perplexity of 10.168289184570312, test perplexity of 10.024857521057129
        # The model neural_lstm_5.pkl best valid perplexity is 10.168289184570312 and test perplexity is 10.024857521057129
    train_and_evaluate_lstm_model(embedding_dim=300, hidden_size=200, num_layers=3, bidirectional=False, dropout=0.35, learning_rate=0.005, batch_size=16, epoches=40, saved_name='neural_lstm_6.pkl', load_path='neural_lstm_6.pkl')


    # model = LSTMModel(dictionary_size=20, embedding_dim=50, hidden_size=200, num_layers=3, batch_size=1, bidirectional=False, dropout=0)
    # print('after create model')
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    # print('after create optimizer')
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # loss_function = nn.CrossEntropyLoss(ignore_index=0)
    #
    # inputs = [[1, 2, 3, 4, 5, 6, 7, 8]]
    # # output = [2, 4, 6, 8, 10, 12, 14, 16]
    # outputs = [1, 2, 3, 4, 5, 6, 7, 8]
    #
    # min_loss = 100
    #
    # for i in range(10000):
    #     model.zero_grad()
    #     model.hidden = model.init_hidden()
    #
    #     batch_log_probs = model.forward(inputs, [8])
    #     print(batch_log_probs.size())
    #     batch_log_probs = batch_log_probs.view(-1, 20)
    #
    #     print(batch_log_probs.data)
    #
    #     _, max_res = torch.max(batch_log_probs, dim=1)
    #     # print('max_res: {}'.format(list(max_res.data)))
    #     # print('target: {}'.format(out))
    #     max_bi = list(zip(max_res.tolist(), outputs))
    #     print('max_bi: {}'.format(max_bi))
    #
    #     loss = loss_function(batch_log_probs, autograd.Variable(torch.LongTensor(outputs)).cuda(gpu_index))
    #     if loss < min_loss:
    #         min_loss = loss
    #     loss.backward()
    #     print('step: {} loss: {}'.format(i, loss))
    #     scheduler.step(loss)
    # print('min_loss: {}'.format(min_loss))






