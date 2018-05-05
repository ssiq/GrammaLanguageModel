from c_code_processer.antlr_preprocess import collect_one_records, collect_dfa_do_parse
from c_code_processer.antlr_util import create_monitor_parser
from common.constants import CACHE_DATA_PATH
from common.util import disk_cache, parallel_map
from read_data.read_experiment_data import read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset

import sys
import os


def split_df_to_part(df, size):
    df_len = len(df)

    df_list = []
    for i in range(0, df_len, size):
        # print(len(df.iloc[i:i+size]))
        df_list += [df.iloc[i:i+size]]
    print(len(df_list))
    return df_list


def read_antlr_parse_records_df(df):
    total = len(df)
    df = df.apply(collect_one_records, raw=True, axis=1, total=total)
    df = df[df['tokens'].map(lambda x: x is not None)]
    return df


@disk_cache(basename='read_antlr_parse_train_records_part', directory=os.path.join(CACHE_DATA_PATH, 'antlr_tmp_parse_records'))
def read_antlr_parse_train_records_part(i):
    train_df, _, _ = read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()
    # df_list = split_df_to_part(train_df, size=100)
    size = 10000
    df = train_df.iloc[i*size:(i+1)*size]
    print('len df: ', len(df))
    # df = read_antlr_parse_records_df(df_list[i])
    df_list = split_df_to_part(df, 1000)
    for df in df_list:
        print('len parallel df: ', len(df))
    res = list(parallel_map(10, read_antlr_parse_records_df, df_list))
    df = res[0]
    for i in range(1, len(res)):
        df = df.append(res[i], ignore_index=True)

    # df = read_antlr_parse_records_df(df)
    return df



@disk_cache(basename='read_antlr_parse_records', directory=os.path.join(CACHE_DATA_PATH, 'antlr_tmp_parse_records'))
def read_antlr_parse_records(df):
    total = len(df)
    df = df.apply(collect_one_records, raw=True, axis=1, total=total)
    return df


def do_apply(df):
    total = len(df)
    df = df.apply(collect_one_records, raw=True, axis=1, total=total)
    df = df[df['tokens'].map(lambda x: x is not None)]
    return df

def process_df_multiple(df):

    df_list = split_df_to_part(df, 1000)

    for df in df_list:
        print('len parallel df: ', len(df))
    res = list(parallel_map(10, do_apply, df_list))
    df = res[0]
    for i in range(1, len(res)):
        df = df.append(res[i], ignore_index=True)
    return df


@disk_cache(basename='read_antlr_parse_records_train_set', directory=CACHE_DATA_PATH)
def read_antlr_parse_records_train_set():
    test_df = read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()[0]
    df = process_df_multiple(test_df)
    print('finish multiple process')
    df = df[df['tokens'].map(lambda x: x is not None)]
    return df


@disk_cache(basename='read_antlr_parse_records_valid_set', directory=CACHE_DATA_PATH)
def read_antlr_parse_records_valid_set():
    valid_df = read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()[1]
    df = process_df_multiple(valid_df)
    df = df[df['tokens'].map(lambda x: x is not None)]
    return df


@disk_cache(basename='read_antlr_parse_records_test_set', directory=CACHE_DATA_PATH)
def read_antlr_parse_records_test_set():
    test_df = read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()[2]
    df = process_df_multiple(test_df)
    df = df[df['tokens'].map(lambda x: x is not None)]
    return df


@disk_cache(basename='read_antlr_parse_records_dataset', directory=CACHE_DATA_PATH)
def read_antlr_parse_records_dataset():
    datasets = read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()
    total = 0
    for df in datasets:
        total += len(df)

    datasets = [process_df_multiple(df) for df in datasets]
    datasets = [df[df['tokens'].map(lambda x: x is not None)] for df in datasets]
    print('train: {}, valid: {}, test: {}'.format(len(datasets[0]), len(datasets[1]), len(datasets[2])))
    return datasets


@disk_cache(basename='read_parser_train_dfa', directory=CACHE_DATA_PATH)
def read_parser_train_dfa():
    train_df = read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()[0]
    train_df['tokens'] = train_df['code'].apply(collect_dfa_do_parse, total=len(train_df))
    tmp_code = '''int main(){
    return 0;
}'''
    _, _, _, parser = create_monitor_parser(tmp_code)
    return parser.decisionsToDFA


def main():
    # for i in range(0, 15):
    #     read_antlr_parse_train_records_part(0)
    # read_antlr_parse_records_dataset()
    read_antlr_parse_records_train_set()
    # read_antlr_parse_records_valid_set()
    # read_antlr_parse_records_test_set()

def main_dfa():
    dfa_list = read_parser_train_dfa()
    tmp_code = '''int main(){
        return 0;
    }'''
    _, _, _, parser = create_monitor_parser(tmp_code)
    print(len(dfa_list))
    for i, dfa in enumerate(dfa_list):
        if dfa is not None:
            if dfa.precedenceDfa:
                print('DFA {} : precedenceDfa is {}'.format(i, dfa.precedenceDfa))
            else:
                print('DFA {} : precedenceDfa is {}: rule: {}'.format(i, dfa.precedenceDfa, dfa.toString(parser.literalNames)))


if __name__ == '__main__':
    # datasets = read_antlr_parse_records_dataset()
    sys.setrecursionlimit(10000)
    main()
