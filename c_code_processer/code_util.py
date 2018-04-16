import re
import collections

import inspect
import typing

from c_code_processer.pycparser import pycparser
import more_itertools
import cytoolz as toolz

from c_code_processer.buffered_clex import BufferedCLex
from common import util


def check_include_between_two_code(code1, code2):
    names1 = extract_include_from_code(code1)
    names2 = extract_include_from_code(code2)
    return equal_include(names1, names2)


def extract_include_from_code(code):
    includes = extract_include(code)
    include_names = [extract_include_name(inc) for inc in includes]
    return include_names


def remove_include(code):
    lines = code.split('\n')
    pattern = re.compile('#include *<(.*)>|#include *"(.*)"')
    lines_without_include = list(filter(lambda line: pattern.match(line) is None, lines))
    return '\n'.join(lines_without_include)


def replace_include_with_blank(code):
    lines = code.split('\n')
    pattern = re.compile('#include *<(.*)>|#include *"(.*)"')
    lines_without_include = [line if pattern.match(line) is None else '' for line in lines]
    return '\n'.join(lines_without_include)


def analyse_include_line_no(code, include_lines):
    lines = code.split('\n')
    include_line_nos = [match_one_include_line_no(lines, include_line) for include_line in include_lines]
    return include_line_nos


def match_one_include_line_no(lines, include_line):
    for i in range(len(lines)):
        if lines[i] == include_line:
            return i
    print('match one include line no error. lines: {}, include_line:{}'.format(lines, include_line))
    return None


def equal_include(names1, names2):
    if len(names1) != len(names2):
        return False
    for inc1, inc2 in zip(names1, names2):
        if inc1 != inc2:
            return False
    return True


def extract_include(code):
    lines = code.split('\n')
    pattern = re.compile('#include *<(.*)>|#include *"(.*)"')
    lines = map(str.strip, lines)
    include_lines = list(filter(lambda line: pattern.match(line) is not None, lines))
    return include_lines


def extract_include_name(include):
    include = include.strip()
    m = re.match('#include *<(.*)>', include)
    if m:
        return m.group(1)
    m = re.match('#include *"(.*)"', include)
    if m:
        return m.group(1)
    return None


def tokenize(code: str):
    lex = BufferedCLex(lambda x1, x2, x3:None, lambda: None, lambda: None, lambda x:None)
    lex.build()
    lex.input(code)
    return lex.tokens_buffer


def create_is_p_fn():
    pattern = re.compile(r"p_.*")

    def is_p_fn(name):
        return pattern.match(name)
    return is_p_fn


class Production(object):
    def __init__(self, left: str, right: typing.List[str], token_id_map: typing.Dict[str, int]):
        self._left = left
        self._right = right
        self._left_id = token_id_map[left]
        self._right_id = [token_id_map[c] for c in right]

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def left_id(self):
        return self._left_id

    @property
    def right_id(self):
        return self._right_id

    def __str__(self):
        return "{}    : {}".format(self.left, " ".join(self.right))

    def __eq__(self, other):
        if not isinstance(other, Production):
            return False
        if self.left != other.left:
            return False
        if self.right != other.right:
            return False
        if self.left_id != other.left_id:
            return False
        if self.right_id != other.right_id:
            return False
        return True

    def __hash__(self):
        return self.left.__hash__() + self.left_id + sum(t.__hash__() for t in self.right) + sum(self.right_id)


class ProductionVocabulary(object):
    def __init__(self,
                 production_list: typing.List,):
        self._token_set = set(i.strip() for i in more_itertools.collapse(production_list))
        self._id_token_map = self._get_set_id_map(self._token_set)
        self._token_id_map = util.reverse_dict(self._id_token_map)
        self._production_list = [Production(left, right, self._token_id_map) for left, right in production_list]
        self._id_production_map = self._get_set_id_map(self._production_list)
        self._production_id_map = util.reverse_dict(self._id_production_map)
        self._token_derivate_map = toolz.groupby(lambda x: x.left_id, self._production_list)
        self._string_production_map = {str(production): production for production in self._production_list}
        self._terminal_set = set(i.strip() for i in pycparser.c_lexer.CLexer.tokens)

    def _get_set_id_map(self, s):
        s = sorted(s, key=lambda x: str(x))
        return dict(enumerate(s))

    def get_production_by_id(self, i):
        return self._id_production_map[i]

    def get_matched_production(self, token_id):
        token = self._id_token_map[token_id]
        return self._token_derivate_map[token]

    def get_production_by_production_string(self, doc):
        """
        :param doc: a production string
        :return: a list of production in the production string
        """
        production_str_list = split_production_string(doc)
        productions = []
        for left, right in production_str_list:
            productions.append(self._string_production_map["{}    : {}".format(left, " ".join(right))])
        return productions

    def __str__(self):
        return "\n".join([str(production) for production in self._production_list])


def split_production_string(s: str):
    left, rights = s.split(":")
    left = left.strip()
    rights = rights.split("|")
    rights = [re.split("\s+", right.strip()) for right in rights]
    lefts = [left] * len(rights)
    productions = list(zip(lefts, rights))
    return productions


def get_all_c99_production_vocabulary():
    is_p_fn = create_is_p_fn()
    parser = pycparser.CParser()
    parse_fn_tuple_list = filter(lambda x: is_p_fn(x[0]) and x[0] != "p_error", inspect.getmembers(parser))
    production_list = map(lambda x: x[1].__doc__, parse_fn_tuple_list)

    production_list = list(more_itertools.flatten(list(map(split_production_string, production_list))))

    production_vocabulary = ProductionVocabulary(production_list,)

    return production_vocabulary


if __name__ == '__main__':
    print(get_all_c99_production_vocabulary())
