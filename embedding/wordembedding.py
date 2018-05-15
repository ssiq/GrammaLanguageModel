import abc

import more_itertools
import numpy
import numpy as np

from common import util
from common.constants import pre_defined_c_tokens_map


class Vocabulary(object):
    def __init__(self,
                 word_set:set,
                 word_to_id_dict: dict,
                 begin_tokens,
                 end_tokens,
                 unk_token,
                 add_position_to_dict=True):
        self.unk = unk_token
        self.begin_tokens = begin_tokens
        self.end_tokens = end_tokens
        if add_position_to_dict:
            position_tokens = set(begin_tokens)
            position_tokens |= set(end_tokens)
            position_tokens |= {unk_token}
            self.word_set = word_set | position_tokens
            for token in sorted(position_tokens):
                word_to_id_dict[token] = len(word_to_id_dict)
        self.word_to_id_dict = word_to_id_dict
        self.id_to_word_dict = util.reverse_dict(self.word_to_id_dict)

    def word_to_id(self, word):
        if word in self.word_to_id_dict.keys():
            return self.word_to_id_dict[word]
        else:
            return self.word_to_id_dict[self.unk]

    def id_to_word(self, i):
        return self.id_to_word_dict[i]

    def parse_text(self, texts, use_position_label = False):
        """
        :param texts: a list of list of token
        :param use_position_label: whether add begin and end token
        :return:
        """
        if use_position_label:
            texts = [self.begin_tokens + text + self.end_tokens for text in texts]
        max_text = max(map(lambda x:len(x), texts))
        texts = [[self.word_to_id(token) for token in text] for text in texts]
        texts = [text+[0]*(max_text-len(text)) for text in texts]
        return texts

    def parse_text_without_pad(self, texts, use_position_label=False):
        if use_position_label:
            texts = [self.begin_tokens + text + self.end_tokens for text in texts]
        texts = [[self.word_to_id(token) for token in text] for text in texts]
        return texts

    @property
    def vocabulary_size(self):
        return len(self.id_to_word_dict)


def load_vocabulary(load_vocabulary_fn, load_vocabulary_id_dict, begin_tokens ,end_tokens, unk_token) -> Vocabulary:
    return Vocabulary(load_vocabulary_fn(), load_vocabulary_id_dict(), begin_tokens, end_tokens, unk_token)

def load_keyword_identifier_split_vocabulary(load_vocabulary_fn, begin_tokens, end_tokens, unk_token):
    keyword_map = pre_defined_c_tokens_map
    keyword_set = sorted(set(keyword_map.values()))
    vocabulary = load_vocabulary_fn()
    vocabulary_map = {}
    for keyword in keyword_set:
        vocabulary_map[keyword] = len(vocabulary_map)
    for t in begin_tokens:
        vocabulary_map[t] = len(vocabulary_map)
    for t in end_tokens:
        vocabulary_map[t] = len(vocabulary_map)
    for t in {"CONSTANT", "STRING_LITERAL"}:
        vocabulary_map[t] = len(vocabulary_map)
    vocabulary_map[unk_token] = len(vocabulary_map)
    keyword_index = len(vocabulary_map)
    for v in vocabulary:
        if v not in vocabulary_map:
            vocabulary_map[v] = len(vocabulary_map)
    return Vocabulary(vocabulary, vocabulary_map, begin_tokens, end_tokens, unk_token, False), keyword_index
