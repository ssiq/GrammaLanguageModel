from collections import Counter

import more_itertools
from toolz.sandbox import unzip

from c_code_processer.code_util import tokenize, MonitoredParser, parse_tree_to_top_down_process
from common.constants import CACHE_DATA_PATH
from common.util import disk_cache, show_process_map
from read_data.read_experiment_data import read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset


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


def read_parsed_top_down_code(debug=False):
    data = read_parsed_tree_code(debug)

    def parse_df(df):
        df['parse_tree'] = show_process_map(parse_tree_to_top_down_process, df['parse_tree'])
        del df['ast']
        return df

    return [parse_df(df) for df in data]


if __name__ == '__main__':
    # for i in read_filtered_without_include_code_tokens():
    #     # print(i[0][:10])
    #     pass
    import sys
    sys.setrecursionlimit(1000000)
    read_parsed_tree_code()
