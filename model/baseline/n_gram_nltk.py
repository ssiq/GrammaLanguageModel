import sys

import torch

from nltk import ngrams

from common.constants import CACHE_DATA_PATH
from common.torch_util import calculate_accuracy_of_code_completion, get_predict_and_target_tokens
from common.util import disk_cache
from read_data.load_parsed_data import read_filtered_without_include_code_tokens

from nltk.model.ngram import LaplaceNgramModel
from nltk.model.counter import build_vocabulary, NgramCounter
import more_itertools

from read_data.read_example_code import read_example_code_tokens

is_debug = False
order = 8
start = '<s>'
end = '</s>'
unk = '<unk>'

ngrams_kwargs = {
        "pad_left": True,
        "pad_right": True,
        "left_pad_symbol": start,
        "right_pad_symbol": end
    }

def read_and_filter_data():
    if is_debug:
        train_data, valid_data, test_data = [d[:100] for d in read_filtered_without_include_code_tokens()]
    else:
        train_data, valid_data, test_data = read_filtered_without_include_code_tokens()
    train_data = list(filter(lambda x: len(x) < 500, train_data))
    valid_data = list(filter(lambda x: len(x) < 500, valid_data))
    test_data = list(filter(lambda x: len(x) < 500, test_data))
    return train_data, valid_data, test_data


def create_ngrams_vocabulary(train_data):
    # print('train_data: ', train_data)
    vocab = build_vocabulary(1, *train_data)
    print('vocabulary length: {}'.format(len(vocab)))
    return vocab


def create_vocabulary_transform_dict(vocab, start, end, unk):
    word_to_id_dict = {word: i for i, word in enumerate(vocab)}
    word_to_id_dict[start] = len(word_to_id_dict.keys())
    word_to_id_dict[end] = len(word_to_id_dict.keys())
    word_to_id_dict[unk] = len(word_to_id_dict.keys())
    id_to_word_dict = {i: word for word, i in word_to_id_dict.items()}
    return word_to_id_dict, id_to_word_dict


def create_dict_fn(work_dict, unk_value):
    def get_item_fn(word):
        if word not in work_dict:
            return unk_value
        return work_dict[word]
    return get_item_fn


def create_counter_and_train_text(train_text_list, vocab=None, order=order):
    if vocab is None:
        vocab = create_ngrams_vocabulary(train_text_list)
    counters = NgramCounter(order, vocab)
    for i, text in enumerate(train_text_list):
        counters.train_counts([text])
        if i % 1 == 0:
            print('in step: {}, len text: {}'.format(i, len(text)))
    return counters


@disk_cache(basename='n_grams_counter', directory=CACHE_DATA_PATH)
def create_trained_counter():
    train_data, valid_data, test_data = read_and_filter_data()
    print('train_data len: {}'.format(len(train_data)))
    vocab = create_ngrams_vocabulary(train_data)
    counter = create_counter_and_train_text(train_data, vocab, order)
    print(counter.ngrams[3][('char', 'trump')][','], counter.ngrams[3][('char', 'trump')].N())
    return counter


@disk_cache(basename='n_grams_counter_test', directory=CACHE_DATA_PATH)
def create_trained_counter_test():
    train_data, valid_data, test_data = read_and_filter_data()
    print('train_data len: {}'.format(len(train_data)))
    vocab = create_ngrams_vocabulary(train_data)
    counter = create_counter_and_train_text(train_data, vocab, order)
    print(counter.ngrams[3][('char', 'trump')][','], counter.ngrams[3][('char', 'trump')].N())
    return counter


# def transform_counter_ctx(ctx_dict):
#     ctx_total = ctx_dict.N()
#     for word in ctx_dict:
#         ctx_dict[word] = ctx_dict[word]/ctx_total
#     return ctx_dict


# def transform_counter_ctx(ctx_dict):
#     return produce_one_token_probility(ctx_dict, vocab_size, word_to_id_fn)
#     ctx_total = ctx_dict.N()
#     for word in ctx_dict:
#         ctx_dict[word] = ctx_dict[word]/ctx_total
#     return ctx_dict


# def transform_counter_order(order_dict):
#     print('before transform counter: {}'.format(len(order_dict)))
#     step = 0
#     for ctx in order_dict:
#         print('transform step: {}'.format(step))
#         step += 1
#         order_dict[ctx] = transform_counter_ctx(order_dict[ctx])
#     print('end transform counter: {}'.format(len(order_dict)))
#     return order_dict


def transform_counter_order(order_dict, vocab_size, word_to_id_fn):
    print('before transform counter: {}'.format(len(order_dict)))
    step = 0
    for ctx in order_dict:
        if 'main' in ctx:
            print(order_dict[ctx])
        if step % 100 == 0:
            print('transform step: {}'.format(step))
            sys.stdout.flush()
            sys.stderr.flush()
        step += 1
        order_dict[ctx] = produce_one_token_probility(order_dict[ctx], vocab_size, word_to_id_fn)
    print('end transform counter: {}'.format(len(order_dict)))
    return order_dict


def evaluate_one_text(counter, test_data, cur_order, word_to_id_fn, vocab_size, is_example=False, id_to_word_fn=None):
    accuracy_dict = {}
    batch_count = 0
    step = 0
    end_id = word_to_id_fn(end)

    no_exist_probs = [1/(vocab_size-2) for i in range(vocab_size-3)]
    no_exist_probs += [0]
    no_exist_probs += [1/(vocab_size-2)]
    no_exist_probs += [0]

    order_dict = transform_counter_order(counter.ngrams[cur_order], vocab_size, word_to_id_fn)

    for text in test_data:
        step += 1
        batch_size = 1
        probs = []
        target = [word_to_id_fn(word) for word in text]
        target += [end_id]
        for ng in ngrams(text, cur_order, **ngrams_kwargs):
            ctx, word = ng[:-1], ng[-1]
            if end in ctx:
                break

            # one_position_prob = produce_one_token_probility(order_dict[ctx], vocab_size, word_to_id_fn)
            one_position_prob = order_dict[ctx]
            if len(one_position_prob) != vocab_size:
                one_position_prob = no_exist_probs
            probs += [one_position_prob]
        if is_example:
            predict_tokens, target_tokens = get_predict_and_target_tokens(torch.Tensor([probs]), [target], id_to_word_fn, k=1)
            i = 0
            for pre, tar in zip(predict_tokens, target_tokens):
                print('{} in step {} predict token: {}'.format(i, step, pre))
                print('{} in step {} target token: {}'.format(i, step, tar))
                i += 1
            continue
        accuracy = calculate_accuracy_of_code_completion(torch.Tensor([probs]), torch.LongTensor([target]))
        batch_count += len(target)
        for key, value in accuracy.items():
            accuracy_dict[key] = accuracy_dict.get(key, 0) + value
        if step % 100 == 0:
            print('in step : {}, total: {}, accuracy: {}'.format(step, len(target), accuracy))
            sys.stdout.flush()
            sys.stderr.flush()
    total_accuracy = {key: value / batch_count for key, value in accuracy_dict.items()}
    return total_accuracy


def produce_one_token_probility(ctx_dict, vocab_size, word_to_id_fn):
    one_position_prob = [0 for i in range(vocab_size)]
    total = ctx_dict.N()
    for one_word in ctx_dict:
        one_position_prob[word_to_id_fn(one_word)] = ctx_dict[one_word]/total
    return one_position_prob


def evaluate_counter(counter, is_example=False):
    train_data, _, test_data = read_and_filter_data()
    vocab = create_ngrams_vocabulary(train_data)
    word_to_id_dict, id_to_word_dict = create_vocabulary_transform_dict(vocab, start, end, unk)
    word_to_id_fn = create_dict_fn(word_to_id_dict, word_to_id_dict[unk])
    id_to_word_fn = create_dict_fn(id_to_word_dict, unk)
    vocab_size = len(word_to_id_dict.keys())

    if is_example:
        example_data = read_example_code_tokens()
        test_data = example_data

    del train_data

    for ord in range(2, order):
        print('before evaluate order: {}'.format(ord))
        accuracy = evaluate_one_text(counter, test_data, ord, word_to_id_fn, vocab_size, is_example, id_to_word_fn)
        print("The order {} accuracy is {}".format(ord, accuracy))


def train_and_evaluate():
    if is_debug:
        counter = create_trained_counter_test()
    else:
        counter = create_trained_counter()
    # train_data, valid_data, test_data = read_and_filter_data()
    # vocab = create_ngrams_vocabulary(train_data)
    # print('has <s>: {}'.format('<s>' in vocab))
    # print(counter.ngrams[3][('char', 'trump')][','], counter.ngrams[3][('char', 'trump')].N())
    evaluate_counter(counter, is_example=True)



if __name__ == '__main__':
    train_and_evaluate()

    # train_data, valid_data, test_data = [d[:100] for d in read_filtered_without_include_code_tokens()]
    # counter = create_counter_and_train_text(train_data)

    # # train_data, valid_data, test_data = read_filtered_without_include_code_tokens()
    # train_data = list(filter(lambda x: len(x) < 500, train_data))
    # valid_data = list(filter(lambda x: len(x) < 500, valid_data))
    # test_data = list(filter(lambda x: len(x) < 500, test_data))
    # print(train_data[0])
    # vocab = build_vocabulary(1, *train_data)
    # print(len(vocab), vocab)
    # # for i in range(0, len(train_data), 1000):
    # order = 3
    # counters = NgramCounter(order, vocab)
    # counters.train_counts([train_data[0]])
    # print(type(counters.ngrams), type(counters.ngrams[3]), type(counters.ngrams[3][('int', 'max')]), type(counters.ngrams[3][('int', 'max')]['=']))
    # print(counters.ngrams[3][('char', 'trump')][','])
    #
    # counters = combine_counter([counters, counters], order)
    # print(counters.ngrams[3][('char', 'trump')][','])

    # for i, text in enumerate(train_data):
    #     # counters = count_ngrams(3, vocab, train_data)
    #     counters.train_counts([text])
    #     if i % 10 == 0:
    #         print('in step: {}, len text: {}'.format(i, len(text)))
    # ctx = counters.ngrams[('int', 'max')]
    # print(counters.unigrams)
    # model = LaplaceNgramModel(counters)
    # res_count = model.ngrams[('int', 'max')]['=']
    # res = model.check_context(('int', 'max'))
    # res_c = model.ngrams[('int', 'max')].freq('=')
    # res_d = model.ngrams[('int', 'max')].N()
    # print(res, res_count, res_c, res_d)
    # print(model.ngrams[('int', 'max')])






    # train_and_evaluate_n_gram_language_model(embedding_dim=100, context_size=3, layer_num=3,
    #                                          hidden_size=100, learning_rate=0.001, batch_size=128,
    #                                          epoches=10, saved_name="neural_n_gram_1.pkl")
    # The model neural_n_gram_1.pkl best valid perplexity is 6.904112339019775 and test perplexity is 6.936566352844238

    # train_and_evaluate_n_gram_language_model(embedding_dim=100, context_size=4, layer_num=3,
    #                                          hidden_size=100, learning_rate=0.001, batch_size=128,
    #                                          epoches=10, saved_name="neural_n_gram_2.pkl")
    # The model neural_n_gram_2.pkl best valid perplexity is 6.278172492980957 and test perplexity is 6.302302837371826

    # train_and_evaluate_n_gram_language_model(embedding_dim=100, context_size=5, layer_num=3,
    #                                          hidden_size=100, learning_rate=0.001, batch_size=128,
    #                                          epoches=10, saved_name="neural_n_gram_3.pkl")
    # The model neural_n_gram_3.pkl best valid perplexity is 5.920135021209717 and test perplexity is 5.940866947174072
    pass
