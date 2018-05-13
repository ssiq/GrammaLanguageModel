from c_code_processer.buffered_clex import BufferedCLex
from common.constants import ROOT_PATH
from common.parse_util import init_pycparser

import os


def extract_identifier(code):
    c_parser = init_pycparser(lexer=BufferedCLex)
    c_parser.parse(code)
    global_scope = c_parser._scope_stack[0]
    ids = list(global_scope.keys())
    print(ids)
    return ids


def extract_fake_c_header_identifier():
    file_list = ['math.h', 'stdio.h', 'stdlib.h', 'string.h']
    res = set()
    for fp in file_list:
        afp = os.path.join(ROOT_PATH, 'c_code_processer', 'fake_c_header', fp)
        with open(afp) as f:
            text = f.read()
            res |= set(extract_identifier(text))
    return list(res)


if __name__ == '__main__':
    res = extract_fake_c_header_identifier()
    print(res)
