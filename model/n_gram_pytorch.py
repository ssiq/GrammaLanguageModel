import os

import more_itertools
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils import shuffle
from more_itertools import windowed
import numpy as np
from toolz.sandbox import unzip
import cytoolz as toolz

from common.util import batch_holder
from embedding.wordembedding import load_vocabulary
from read_data.load_parsed_data import read_filtered_without_include_code_tokens, get_token_vocabulary, \
    get_vocabulary_id_map
import config
from common import util, torch_util

gpu_index = 1
BEGIN, END, UNK = ["<BEGIN>", "<END>", "<UNK>"]

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, batch_size, output_layer_num, hidden_size):
        super(NGramLanguageModeler, self).__init__()
        self._batch_size = batch_size
        self._context_size = context_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim).cpu()
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size).cuda(gpu_index)
        self.linear2 = [nn.Linear(hidden_size, hidden_size).cuda(gpu_index) for _ in range(output_layer_num-2)]
        self.linear3 = nn.Linear(hidden_size, vocab_size).cuda(gpu_index)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((self._batch_size,  -1)).cuda(gpu_index)
        out = F.relu(self.linear1(embeds))
        for lin in self.linear2:
            out = F.relu(lin(out))
        out = self.linear3(out)
        return out

@toolz.curry
def parse_xy(codes, context_size, token_id_map_fn):
    previous_tokens = [BEGIN+str(i) for i in range(context_size)]
    end_tokens = [END]
    codes = [previous_tokens + code + end_tokens for code in codes]
    codes = [[token_id_map_fn(token) for token in code] for code in codes]
    res = []
    for tokens in codes:
        res.extend(list(windowed(tokens, context_size+1)))

    X = []
    y = []
    for a, b in [(xy[:-1], xy[-1]) for xy in res]:
        X.append(a)
        y.append(b)

    return X, y

def trian(X, batch_size, loss_function, model, optimizer, y):
    steps = 0
    total_loss = torch.Tensor([0])
    X, y = shuffle(X, y)
    # print("X.shape:{}".format(np.array(X).shape))
    for context_idxs, target_idxs in batch_holder(X, y, batch_size=batch_size)():
        if len(context_idxs) != batch_size:
            break
        # print("context_id_shape:{}".format(np.array(context_idxs).shape))
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        model.zero_grad()

        log_probs = model.forward(context_var)

        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor(target_idxs)).cuda(gpu_index))

        loss.backward()
        optimizer.step()

        total_loss += loss.data.cpu()

        steps += 1

    return total_loss/steps

def evaluate(model, X, y, batch_size, loss_function):
    steps = 0
    total_loss = torch.Tensor([0])
    for context_idxs, target_idxs in batch_holder(X, y, batch_size=batch_size)():
        if len(context_idxs) != batch_size:
            break
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        log_probs = model.forward(context_var)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor(target_idxs)).cuda(gpu_index))
        total_loss += loss.data.cpu()
        steps += 1
    return total_loss / steps


def train_and_evaluate_n_gram_language_model(embedding_dim,
                                             context_size,
                                             layer_num,
                                             hidden_size,
                                             learning_rate,
                                             batch_size,
                                             epoches,
                                             saved_name,
                                             ):
    debug = False
    save_path = os.path.join(config.save_model_root, saved_name)

    begin_tokens = [BEGIN+str(i) for i in range(context_size)]
    vocabulary = load_vocabulary(get_token_vocabulary,
                                 get_vocabulary_id_map,
                                 begin_tokens,
                                 [END],
                                 UNK)
    vocabulary_size = vocabulary.vocabulary_size
    print("The vocabulary_size:{}".format(vocabulary_size))

    if debug:
        train_data, valid_data, test_data = [d[:100] for d in read_filtered_without_include_code_tokens()]
    else:
        train_data, valid_data, test_data = read_filtered_without_include_code_tokens()

    print("train data size:{}".format(len(train_data)))

    loss_function = nn.CrossEntropyLoss()
    model = NGramLanguageModeler(vocabulary_size, embedding_dim, context_size, batch_size, layer_num, hidden_size)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train_X, train_y = parse_xy(train_data, context_size, vocabulary.word_to_id)
    valid_X, valid_y = parse_xy(valid_data, context_size, vocabulary.word_to_id)
    test_X, test_y = parse_xy(test_data, context_size, vocabulary.word_to_id)
    best_valid_perplexity = None
    best_test_perplexity = None
    for epoch in range(epoches):
        train_loss = trian(train_X, batch_size, loss_function, model, optimizer, train_y)
        valid_loss = evaluate(model, valid_X, valid_y, batch_size, loss_function)
        test_loss = evaluate(model, test_X, test_y, batch_size, loss_function)

        train_perplexity = torch.exp(train_loss)[0]
        valid_perplexity = torch.exp(valid_loss)[0]
        test_perplexity = torch.exp(test_loss)[0]

        # print("valid_perplexity:{}".format(valid_perplexity))
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
    train_and_evaluate_n_gram_language_model(embedding_dim=100, context_size=3, layer_num=3,
                                             hidden_size=100, learning_rate=0.001, batch_size=128,
                                             epoches=10, saved_name="neural_n_gram_1.pkl")
    train_and_evaluate_n_gram_language_model(embedding_dim=100, context_size=4, layer_num=3,
                                             hidden_size=100, learning_rate=0.001, batch_size=128,
                                             epoches=10, saved_name="neural_n_gram_2.pkl")
    train_and_evaluate_n_gram_language_model(embedding_dim=100, context_size=5, layer_num=3,
                                             hidden_size=100, learning_rate=0.001, batch_size=128,
                                             epoches=10, saved_name="neural_n_gram_3.pkl")
