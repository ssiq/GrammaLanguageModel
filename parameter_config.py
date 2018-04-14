from collections import namedtuple

from embedding.character_embedding import load_character_vocabulary
from embedding.wordembedding import load_vocabulary
from model.rnn_lm import rnn_model

import cytoolz as toolz
import tensorflow as tf

embedding_object_fn = namedtuple("Embedding", ["parse_fn", "create_embedding_layer_fn"])

class EmbeddingFunc(object):
    def __init__(self,
                 fn,
                 vocabulary_size):
        self._fn = fn
        self._vocabulary_size = vocabulary_size

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __len__(self):
        return self._vocabulary_size

@toolz.curry
def create_word_embedding(data, use_position_label=False, word_vector_name="glove"):
    word_embedding = load_vocabulary(word_vector_name, data, use_position_label)
    parse_fn = lambda texts: [word_embedding.parse_text_without_pad(texts), [len(t) + use_position_label*2 for t in texts]]
    return embedding_object_fn(parse_fn=parse_fn,
                               create_embedding_layer_fn=EmbeddingFunc(
                                   word_embedding.create_embedding_layer,
                                   word_embedding.vocabulary_size))

@toolz.curry
def create_word_and_character_embedding(data, use_position_label=False):
    word_embedding = load_vocabulary("glove", data, use_position_label)
    character_embedding = load_character_vocabulary("bigru", 3, 50, data)
    def parse_fn(texts):
        return [
            word_embedding.parse_text_without_pad(texts),
            [len(t) + use_position_label*2 for t in texts],
            character_embedding.parse_string_without_padding(texts),
            [[len(i_t)+2 for i_t in t] for t in texts]
        ]

    def create_embedding_layer_fn():
        def embedding_fn(words_tensor, character_tensor, character_length):
            word_embedding_fn = word_embedding.create_embedding_layer()
            charcter_embedding_fn = character_embedding.create_embedding_layer()
            w_em = word_embedding_fn(words_tensor)
            c_em = charcter_embedding_fn(character_tensor, character_length)
            return tf.concat((w_em, c_em), axis=-1)
        return embedding_fn

    return embedding_object_fn(parse_fn=parse_fn,
                               create_embedding_layer_fn=EmbeddingFunc(
                                   create_embedding_layer_fn,
                                   word_embedding.vocabulary_size))


config1 = {
    "model_fn": rnn_model,
    "model_parameter": {
        "rnn_layer_num": 3,
        "use_chacacter_embedding": False,
        "hidden_state_size": 150,
        "classes": 6,
        "learning_rate": 0.0005,
        "dropout": True,
        "keep_prob": 0.5,
        "l2": True,
        "l2_decay": 0.01,
        "output_dense_num": 5,
        "output_dense_size": 150,
        "decay_rate": 1.0,
        "decay_steps": 500,
        "output_act": "relu",
        "lm_loss": (0.01, 10),
    },
    "tokenize_name": "stanford",
    "base_name": "config5",
    "create_embedding_fn": create_word_embedding(use_position_label=True, word_vector_name="fasttext"),
    "batch_size": 32,
}