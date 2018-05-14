import sys
from collections import Counter

import more_itertools
from toolz.sandbox import unzip

from c_code_processer.buffered_clex import BufferedCLex
from c_code_processer.code_util import tokenize, MonitoredParser, parse_tree_to_top_down_process, extract_include, \
    replace_include_with_blank
from c_code_processer.fake_c_header.extract_identifier import extract_fake_c_header_identifier
from c_code_processer.slk_parser import slk_parse, c99_slk_parse, monitored_slk_parse
from common.constants import CACHE_DATA_PATH, pre_defined_c_tokens, pre_defined_c_tokens_map
from common.constants import CACHE_DATA_PATH
from common.dataset_util import create_error_tokens_by_operations, create_error_tokens
from common.parse_util import tokenize_by_clex_fn
from common.util import disk_cache, show_process_map
from read_data.read_experiment_data import read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset, \
    read_fake_random_c_error_dataset, read_fake_common_c_error_dataset


def parse_c99_code_to_token(df):
    '''
    :param df:  source code is contained in the df['code']
    :return: a list of token list
    '''
    print_step = 1000
    res = []
    size = len(df['code'])
    for i, code in enumerate(df['code']):
        if i % print_step == 0:
            print("{}/{} finished".format(i, size))
        tmp_res = []
        for token in tokenize(code):
            if token[0].type in {'INT_CONST_DEC', 'INT_CONST_OCT', 'INT_CONST_HEX', 'INT_CONST_BIN',
                            'FLOAT_CONST', 'HEX_FLOAT_CONST',
                            'CHAR_CONST',
                            'WCHAR_CONST',
                            'STRING_LITERAL',
                            'WSTRING_LITERAL', }:
                tmp_res.append(token[0].type)
            else:
                tmp_res.append(token[0].value)
        res.append(tmp_res)
    return res


def parse_c99_code_with_production(df):
    pass


@disk_cache(basename='read_filtered_without_include_code_tokens_moved_literal',
            directory=CACHE_DATA_PATH)
def read_filtered_without_include_code_tokens():
    return [parse_c99_code_to_token(df) for df in read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()]


@disk_cache(basename='get_token_vocabulary_moved_literal', directory=CACHE_DATA_PATH)
def get_token_vocabulary():
    train, _, _ = read_filtered_without_include_code_tokens()
    print(sorted(list(Counter(more_itertools.flatten(train)).items()), key=lambda x:x[1])[:10])

    return set(more_itertools.flatten(train))

@disk_cache(basename='get_vocabulary_id_map_moved_literal', directory=CACHE_DATA_PATH)
def get_vocabulary_id_map():
    word_list = sorted(get_token_vocabulary())
    return {word: i for i, word in enumerate(word_list)}


@disk_cache(basename='get_vocabulary_id_map_with_keyword_moved_literal3', directory=CACHE_DATA_PATH)
def get_vocabulary_id_map_with_keyword():
    p_dict ={"ID": 1,
                             "CONSTANT": 2, # all constants
                             "STRING_LITERAL": 3, # all string literal
                             "LPAREN": 4,
                             "RPAREN": 5,
                             "LBRACKET": 6,
                             "RBRACKET": 7,
                             "PERIOD": 8,
                             "ARROW": 9,
                             "PLUSPLUS": 10,
                             "MINUSMINUS": 11,
                             "COMMA": 12,
                             "SIZEOF": 13,
                             "AND": 14,
                             "TIMES": 15,
                             "PLUS": 16,
                             "MINUS": 17,
                             "NOT": 18,
                             "LNOT": 19,
                             "DIVIDE": 20,
                             "MOD": 21, "LSHIFT": 22, "RSHIFT": 23,
                             "LT": 24, "GT": 25, "LE": 26, "GE": 27, "EQ": 28, "NE": 29,
                             "XOR": 30, "OR": 31, "LAND": 32, "LOR": 33, "CONDOP": 34, "COLON": 35,
                             "EQUALS": 36, "TIMESEQUAL": 37, "DIVEQUAL": 38, "MODEQUAL": 39, "PLUSEQUAL": 40,
                             "MINUSEQUAL": 41, "LSHIFTEQUAL": 42, "RSHIFTEQUAL": 43, "ANDEQUAL": 44,
                             "XOREQUAL": 45, "OREQUAL": 46, "SEMI": 47, "TYPEDEF": 48, "EXTERN": 49,
                             "STATIC": 50, "AUTO": 51, "REGISTER": 52, "VOID": 53, "CHAR": 54, "SHORT": 55,
                             "INT": 56, "LONG": 57, "FLOAT": 58, "DOUBLE": 59, "SIGNED": 60, "UNSIGNED": 61,
                             "_BOOL": 62, "_COMPLEX": 63, "IMAGINARY_": 64, "TYPEID": 65, "LBRACE": 66,
                             "RBRACE": 67, "STRUCT": 68, "UNION": 69, "ENUM": 70, "CONST": 71, "RESTRICT": 72,
                             "VOLATILE": 73, "INLINE": 74, "ELLIPSIS": 75, "CASE": 76, "DEFAULT": 77, "IF": 78,
                             "SWITCH": 79, "ELSE": 80, "FOR": 81, "WHILE": 82, "DO": 83, "GOTO": 84,
                             "CONTINUE": 85, "BREAK": 86, "RETURN": 87, "END_OF_SLK_INPUT": 88,
                             }
    keyword_id_set = sorted(set(pre_defined_c_tokens) | {"CONSTANT", "STRING_LITERAL"})
    pre_defined_c_tokens_map_key = pre_defined_c_tokens_map.keys()
    keyword_id_map = {}
    index = 0
    for k in p_dict.keys():
        if k in pre_defined_c_tokens_map_key:
            keyword_id_map[pre_defined_c_tokens_map[k]] = index
            index += 1
        elif k in {"CONSTANT", "STRING_LITERAL"}:
            keyword_id_map[k] = index
            index += 1
    word_list = sorted(set(get_token_vocabulary()) - set(keyword_id_set))
    print("identifier index begin:{}".format(index))
    word_map = {word: i for i, word in enumerate(word_list, start=len(keyword_id_map))}
    return {**keyword_id_map, **word_map}


# @disk_cache(basename="read_parsed_tree_code", directory=CACHE_DATA_PATH)
def read_parsed_tree_code(debug=False):
    def parse_df(df):
        monitor = MonitoredParser()
        parsed_code = show_process_map(monitor.parse_get_production_list_and_token_list, df['code'],
                                       error_default_value=(None, None, None))
        parsed_code = unzip(parsed_code)
        df['parse_tree'] = list(parsed_code[0])
        df['ast'] = list(parsed_code[1])
        df['tokens'] = list(parsed_code[2])
        return df
    if not debug:
        return [parse_df(df) for df in read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()]
    else:
        return [parse_df(df.head(100)) for df in read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()]


@disk_cache(basename="read_parsed_top_down_code_test", directory=CACHE_DATA_PATH)
def read_parsed_top_down_code(debug=False):
    data = read_parsed_tree_code(debug)

    def parse_df(df):
        df['parse_tree'] = show_process_map(parse_tree_to_top_down_process, df['parse_tree'])
        del df['ast']
        return df

    return [parse_df(df) for df in data]


@disk_cache(basename="read_parsed_slk_top_down_code", directory=CACHE_DATA_PATH)
def read_parsed_slk_top_down_code(debug=False):
    def parse_df(df):
        clex = BufferedCLex(error_func=lambda self, msg, line, column: None,
                            on_lbrace_func=lambda: None,
                            on_rbrace_func=lambda: None,
                            type_lookup_func=lambda typ: None)
        clex.build()
        parse_fn = slk_parse(clex=clex)
        parsed_code = show_process_map(parse_fn, df['code'],
                                       error_default_value=(None, None))
        parsed_code = unzip(parsed_code)
        df['parse_tree'] = list(parsed_code[0])
        df['tokens'] = list(parsed_code[1])
        return df
    if not debug:
        return [parse_df(df) for df in read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()]
    else:
        return [parse_df(df.head(100)) for df in read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()]

@disk_cache(basename="read_parsed_c99_slk_top_down_code2", directory=CACHE_DATA_PATH)
def read_parsed_c99_slk_top_down_code(debug=False):
    def parse_df(df):
        clex = BufferedCLex(error_func=lambda self, msg, line, column: None,
                            on_lbrace_func=lambda: None,
                            on_rbrace_func=lambda: None,
                            type_lookup_func=lambda typ: None)
        clex.build()
        parse_fn = c99_slk_parse(clex=clex)
        parsed_code = show_process_map(parse_fn, df['code'],
                                       error_default_value=(None, None))
        parsed_code = unzip(parsed_code)
        df['parse_tree'] = list(parsed_code[0])
        df['tokens'] = list(parsed_code[1])
        return df
    if not debug:
        return [parse_df(df) for df in read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()]
    else:
        return [parse_df(df.head(100)) for df in read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()]

@disk_cache(basename="read_monitored_parsed_c99_slk_top_down_code2", directory=CACHE_DATA_PATH)
def read_monitored_parsed_c99_slk_top_down_code(debug=False):
    def parse_df(df):
        identifier_set, type_set = extract_fake_c_header_identifier()
        clex = BufferedCLex(error_func=lambda self, msg, line, column: None,
                            on_lbrace_func=lambda: None,
                            on_rbrace_func=lambda: None,
                            type_lookup_func=lambda typ: None)
        clex.build()
        BEGIN, END, UNK = ["<BEGIN>", "<END>", "<UNK>"]
        from embedding.wordembedding import load_vocabulary
        vocabulary = load_vocabulary(get_token_vocabulary, get_vocabulary_id_map_with_keyword, [BEGIN], [END], UNK)
        print("the size of predefined_identifer:{}".format(len(identifier_set)))
        print("the size of typeset:{}".format(len(type_set)))
        parse_fn = monitored_slk_parse(clex=clex, predefined_identifer=identifier_set, predefined_typename=type_set,
                                       vocabulary=vocabulary)
        parsed_code = show_process_map(parse_fn, df['code'],
                                       error_default_value=tuple([None, ] * 7))
        parsed_code = unzip(parsed_code)
        df['parse_tree'] = list(parsed_code[0])
        df['tokens'] = list(parsed_code[1])
        df['consistent_identifier'] = list(parsed_code[2])
        df['identifier_scope_index'] = list(parsed_code[3])
        df['is_identifier'] = list(parsed_code[4])
        df['max_scope_list'] = list(parsed_code[5])
        df['consistent_typename'] = list(parsed_code[6])
        return df

    if not debug:
        return [parse_df(df) for df in read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()]
    else:
        return [parse_df(df.head(100)) for df in read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()]


@disk_cache(basename="read_random_error_c99_code_tokens_list", directory=CACHE_DATA_PATH)
def read_random_error_c99_code_tokens_list():
    print('in read_random_error_c99_code_tokens_list')
    def parse_df_to_token_list(df):
        print('start parse df to token list: {}'.format(len(df)))
        similar_code_without_includes = df['similar_code'].map(replace_include_with_blank)
        print('after remove include')
        tokenize_fn = tokenize_by_clex_fn()
        ac_tokens = similar_code_without_includes.map(tokenize_fn)
        print('after tokenize ac code')
        tokens_list = [[tok.value for tok in tokens] for tokens in ac_tokens]
        return tokens_list
    return [parse_df_to_token_list(df) for df in read_fake_random_c_error_dataset()]


@disk_cache(basename="get_random_error_c99_code_token_vocabulary", directory=CACHE_DATA_PATH)
def get_random_error_c99_code_token_vocabulary():
    print('in get_random_error_c99_code_token_vocabulary')
    train, _, _ = read_random_error_c99_code_tokens_list()
    print(sorted(list(Counter(more_itertools.flatten(train)).items()), key=lambda x: x)[:10])

    return set(more_itertools.flatten(train))


@disk_cache(basename="get_random_error_c99_code_token_vocabulary_id_map", directory=CACHE_DATA_PATH)
def get_random_error_c99_code_token_vocabulary_id_map():
    print('in get_random_error_c99_code_token_vocabulary_id_map')
    word_list = sorted(get_random_error_c99_code_token_vocabulary())
    return {word: i for i, word in enumerate(word_list)}


@disk_cache(basename="read_common_error_c99_code_tokens_list", directory=CACHE_DATA_PATH)
def read_common_error_c99_code_tokens_list():
    print('in read_common_error_c99_code_tokens_list')
    def parse_df_to_token_list(df):
        print('start parse df to token list: {}'.format(len(df)))
        similar_code_without_includes = df['similar_code'].map(replace_include_with_blank)
        print('after remove include')
        tokenize_fn = tokenize_by_clex_fn()
        ac_tokens = similar_code_without_includes.map(tokenize_fn)
        print('after tokenize ac code')
        tokens_list = [[tok.value for tok in tokens] for tokens in ac_tokens]
        return tokens_list
    return [parse_df_to_token_list(df) for df in read_fake_common_c_error_dataset()]


@disk_cache(basename="get_common_error_c99_code_token_vocabulary", directory=CACHE_DATA_PATH)
def get_common_error_c99_code_token_vocabulary():
    print('in get_common_error_c99_code_token_vocabulary')
    train, _, _ = read_common_error_c99_code_tokens_list()
    print(sorted(list(Counter(more_itertools.flatten(train)).items()), key=lambda x: x)[:10])

    return set(more_itertools.flatten(train))


@disk_cache(basename="get_common_error_c99_code_token_vocabulary_id_map", directory=CACHE_DATA_PATH)
def get_common_error_c99_code_token_vocabulary_id_map():
    print('in get_common_error_c99_code_token_vocabulary_id_map')
    word_list = sorted(get_common_error_c99_code_token_vocabulary())
    return {word: i for i, word in enumerate(word_list)}


def generate_tokens_for_c_error_dataset(data):
    return [create_error_tokens(df) for df in data]


if __name__ == '__main__':
    # for i in read_filtered_without_include_code_tokens():
    #     # print(i[0][:10])
    #     pass
    # import sys
    # sys.setrecursionlimit(1000000)
    # read_parsed_tree_code()
    # read_parsed_c99_slk_top_down_code()
    # read_random_error_c99_code_tokens_list()
    # get_random_error_c99_code_token_vocabulary()
    # res = get_random_error_c99_code_token_vocabulary_id_map()
    # read_common_error_c99_code_tokens_list()
    # get_common_error_c99_code_token_vocabulary()
    # res = get_common_error_c99_code_token_vocabulary_id_map()
    # print(len(res.keys()))
    # read_parsed_c99_slk_top_down_code()
    # print("next")
    read_monitored_parsed_c99_slk_top_down_code()
    # print("ddd")
    # print(sys.stdin.readline())