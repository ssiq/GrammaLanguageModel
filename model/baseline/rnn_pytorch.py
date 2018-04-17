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


gpu_index = 1
BEGIN, END, UNK = ["<BEGIN>", "<END>", "<UNK>"]


class LSTMModel(nn.Module):

    def __init__(self, dictionary_size, embedding_dim, hidden_size, num_layers, batch_size, bidirectional=False, dropout=0):
        super(LSTMModel, self).__init__()
        self.dictionary_size = dictionary_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        print('dictionary_size: {}, embedding_dim: {}, hidden_size: {}, num_layers: {}, batch_size: {}, bidirectional: {}, dropout: {}'.format(
            dictionary_size, embedding_dim, hidden_size, num_layers, batch_size, bidirectional, dropout))

        self.bidirectional_num = 2 if bidirectional else 1
        self.dropout = dropout


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

        # print('input_size: ', inputs.size())

        embeds = self.word_embeddings(autograd.Variable(inputs).cuda(gpu_index)).view(self.batch_size, -1, self.embedding_dim).cuda(gpu_index)
        # print('embeds_size: {}, embeds is cuda: {}'.format(embeds.size(), embeds.is_cuda))
        embeds = embeds.view(self.batch_size, -1, self.embedding_dim)
        # print('embeds_size: {}, embeds is cuda: {}'.format(embeds.size(), embeds.is_cuda))
        # print('after embeds token_length: {}'.format(token_lengths))
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(embeds, token_lengths, batch_first=True)
        # print('packed_inputs batch size: ', len(packed_inputs.batch_sizes))
        # print('packed_inputs is cuda: {}'.format(packed_inputs.data.is_cuda))
        lstm_out, self.hidden = self.lstm(packed_inputs, self.hidden)
        # print('lstm_out batch size: ', len(lstm_out.batch_sizes))
        # print('lstm_out is cuda: ', lstm_out.data.is_cuda)
        packed_output = nn.utils.rnn.PackedSequence(self.hidden2tag(lstm_out.data).cuda(gpu_index), lstm_out.batch_sizes)    # output shape: [batch_size, token_length, dictionary_size]
        # print('packed_output batch size: ', len(packed_output.batch_sizes))
        # print('packed_output is cuda: ', packed_output.data.is_cuda)

        unpacked_out, unpacked_length = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, padding_value=-1)
        # print('unpacked_out: {}, unpacked_length: {}'.format(unpacked_out.size(), unpacked_length))
        unpacked_out = torch.index_select(unpacked_out, 0, autograd.Variable(idx_sort).cuda(gpu_index))
        # print('unsort unpacked_out: {}'.format(unpacked_out.size()))
        # print('unsort unpacked_out is cuda: {}'.format(unpacked_out.is_cuda))

        return unpacked_out


def train(model, X, y, optimizer, loss_function, batch_size):
    # print('in train')
    steps = 0
    total_loss = torch.Tensor([0])
    print('before shuffle')
    X, y = shuffle(X, y)
    print('finish shuffle: x: {}, y: {}'.format(len(X), len(y)))

    for inp, out in batch_holder(X, y, batch_size=batch_size)():
        if len(inp) != batch_size:
            break

        # print('in one batch: X: {},{}, y: {},{}'.format(len(inp), len(inp[0]), len(out), len(out[0])))
        # print('X size: ', torch.Tensor(inp).size())
        # print('y size: ', torch.Tensor(y).size())
        inp, inp_len = list(zip(*inp))
        # print(type(inp), type(inp[0]))
        inp = util.padded(list(inp), deepcopy=True, fill_value=0)
        out = util.padded(list(out), deepcopy=True, fill_value=-1)
        # print('inp[0]: {}, inp[1]: {}, inp[2]: {}, inp[3]: {}'.format(len(inp[0]), len(inp[1]), len(inp[2]), len(inp[3])))
        # print('in one batch: X: {},{}, y: {},{}'.format(len(inp), len(inp[0]), len(out), len(out[0])))
        # print('X size: ', torch.Tensor(inp).size())
        # print('y size: ', torch.Tensor(out).size())

        model.zero_grad()

        model.hidden = model.init_hidden()

        log_probs = model.forward(inp, inp_len)

        batch_log_probs = log_probs.view(-1, list(log_probs.size())[-1])
        out = list(more_itertools.flatten(out))

        loss = loss_function(batch_log_probs, autograd.Variable(torch.LongTensor(out)).cuda(gpu_index))
        print('step {} loss: {}'.format(steps, loss.data))
        total_loss += loss.data.cpu()

        loss.backward()
        optimizer.step()

        if steps % 1000 == 0:
            sys.stdout.flush()
            sys.stderr.flush()

        steps += 1
    return total_loss / steps


def evaluate(model, X, y, loss_function, batch_size):
    # print('in evaluate')
    steps = 0
    total_loss = torch.Tensor([0])

    for inp, out in batch_holder(X, y, batch_size=batch_size)():
        if len(inp) != batch_size:
            break

        # print('in one batch: X: {},{}, y: {},{}'.format(len(inp), len(inp[0]), len(out), len(out[0])))
        # print('X size: ', torch.Tensor(inp).size())
        # print('y size: ', torch.Tensor(y).size())
        inp, inp_len = list(zip(*inp))
        # print(type(inp), type(inp[0]))
        inp = util.padded(list(inp), deepcopy=True, fill_value=0)
        out = util.padded(list(out), deepcopy=True, fill_value=-1)
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
        total_loss += loss.data.cpu()

        if steps % 1000 == 0:
            sys.stdout.flush()
            sys.stderr.flush()

        steps += 1
    return total_loss / steps


def parse_xy(codes, word_to_id_fn):
    begin_tokens = [BEGIN]
    end_tokens = [END]

    codes = [begin_tokens + code + end_tokens for code in codes]
    codes = [[word_to_id_fn(token) for token in code] for code in codes]
    codes = list(filter(lambda x: len(x) < 500, codes))

    X = []
    y = []
    for code in codes:
        inp = code[:-1]
        out = code[1:]
        X += [(inp, len(inp))]
        y += [out]
    return X, y


def train_and_evaluate_lstm_model(embedding_dim, hidden_size, num_layers, bidirectional, dropout, learning_rate, batch_size, epoches, saved_name):
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
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    print('before create model')
    model = LSTMModel(vocabulary_size, embedding_dim, hidden_size, num_layers, batch_size, bidirectional, dropout)
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

        # print("epoch {}: train perplexity of {},  valid perplexity of {}, test perplexity of {}".
        #       format(epoch, train_perplexity, valid_perplexity, test_perplexity))

        print("epoch {}: train perplexity of {},  valid perplexity of {}, test perplexity of {}".
              format(i, train_perplexity, valid_perplexity, test_perplexity))
        sys.stdout.flush()
        sys.stderr.flush()
    print("The model {} best valid perplexity is {} and test perplexity is {}".
          format(saved_name, best_valid_perplexity, best_test_perplexity))



if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    print('start')

    train_and_evaluate_lstm_model(embedding_dim=100, hidden_size=100, num_layers=1, bidirectional=False, dropout=0, learning_rate=0.01, batch_size=16, epoches=15, saved_name='neural_lstm_1.pkl')
    train_and_evaluate_lstm_model(embedding_dim=300, hidden_size=100, num_layers=1, bidirectional=False, dropout=0, learning_rate=0.01, batch_size=16, epoches=15, saved_name='neural_lstm_2.pkl')
    train_and_evaluate_lstm_model(embedding_dim=100, hidden_size=200, num_layers=1, bidirectional=False, dropout=0, learning_rate=0.01, batch_size=16, epoches=15, saved_name='neural_lstm_3.pkl')
    train_and_evaluate_lstm_model(embedding_dim=100, hidden_size=100, num_layers=2, bidirectional=False, dropout=0, learning_rate=0.01, batch_size=16, epoches=15, saved_name='neural_lstm_4.pkl')






