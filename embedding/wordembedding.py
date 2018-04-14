import abc

import more_itertools
import numpy
import numpy as np
import tensorflow as tf
from gensim.models.fasttext import FastText

import config
from common import util


class WordEmbedding(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model):
        self.model = model

    def __getitem__(self, item):
        # if item in self.model:
        #     return self.model[item]
        # else:
        #     return numpy.random.randn(*self.model['office'].shape)
        try:
            r = self.model[item]
        except Exception:
            r =  numpy.random.randn(*self.model['office'].shape)
        return r


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model


class GloveWordEmbedding(WordEmbedding):
    def __init__(self):
        super().__init__(loadGloveModel(config.pretrained_glove_path))


class FastTextWordEmbedding(WordEmbedding):
    def __init__(self):
        super().__init__(FastText.load_fasttext_format(config.pretrained_fasttext_path))
        print("the fasttext model loaded")


class Vocabulary(object):
    def __init__(self, embedding: WordEmbedding, word_set: set, use_position_label: bool):
        self.unk = '<unk>'
        self.begin = '<BEGIN>'
        self.end = '<END>'
        self.use_position_label = use_position_label
        word_set = sorted(set(word_set))
        self.id_to_word_dict = dict(list(enumerate(word_set, start=1)))
        self.id_to_word_dict[0] = self.unk
        if use_position_label:
            self.id_to_word_dict[len(self.id_to_word_dict)] = self.begin
            self.id_to_word_dict[len(self.id_to_word_dict)] = self.end
        self.word_to_id_dict = util.reverse_dict(self.id_to_word_dict)
        print("The word vocabulary has {} words".format(len(self.word_to_id_dict)))
        self._embedding_matrix = np.array([embedding[b] for a, b in sorted(self.id_to_word_dict.items(), key=lambda x:x[0])])

    def word_to_id(self, word):
        if word in self.word_to_id_dict.keys():
            return self.word_to_id_dict[word]
        else:
            return 0

    def id_to_word(self, i):
        if i:
            return self.id_to_word_dict[i]
        else:
            return self.unk

    @property
    def embedding_matrix(self):
        return self._embedding_matrix

    def create_embedding_layer(self):
        with tf.variable_scope("word_embedding"):
            _tf_embedding = tf.Variable(name="embedding", initial_value=self._embedding_matrix,
                                             dtype=tf.float32, trainable=False)
        def embedding_layer(input_op):
            """
            :param input_op: a tensorflow tensor with shape [batch, max_length] and type tf.int32
            :return: a looked tensor with shape [batch, max_length, embedding_size]
            """
            output = tf.nn.embedding_lookup(_tf_embedding, input_op)
            print("word embedding:{}".format(output))
            return output

        return embedding_layer

    def parse_text(self, texts):
        """
        :param texts: a list of list of token
        :return:
        """
        if self.use_position_label:
            texts = [[self.begin] + text + [self.end] for text in texts]
        max_text = max(map(lambda x:len(x), texts))
        texts = [[self.word_to_id(token) for token in text] for text in texts]
        texts = [text+[0]*(max_text-len(text)) for text in texts]
        return texts

    def parse_text_without_pad(self, texts):
        if self.use_position_label:
            texts = [[self.begin] + text + [self.end] for text in texts]
        texts = [[self.word_to_id(token) for token in text] for text in texts]
        return texts

    @property
    def vocabulary_size(self):
        return len(self.id_to_word_dict)


def load_vocabulary(word_vector_name, text_list, use_position_label=False) -> Vocabulary:
    namd_embedding_dict = {"glove": GloveWordEmbedding, "fasttext": FastTextWordEmbedding}
    word_set = more_itertools.collapse(text_list)
    return Vocabulary(namd_embedding_dict[word_vector_name](), word_set, use_position_label=use_position_label)
