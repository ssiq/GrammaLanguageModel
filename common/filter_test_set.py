import pandas as pd
import copy

from common.util import group_df_to_grouped_list


def filter_distinct_table_key(data_df, key, max_num=None):
    if max_num is None:
        max_num = float('inf')
    group_list = group_df_to_grouped_list(data_df, key)
    print('group_list', len(group_list))
    num = min(max_num, len(group_list[0]))
    group_res = group_list[0].sample(num).copy(deep=True)
    i = 0
    for group in group_list[1:]:
        print('filter_distinct_table_key: {} in {}'.format(i, len(group_list)))
        i += 1
        num = min(max_num, len(group))
        group_res = group_res.append(group.sample(num), ignore_index=True)
    return group_res


def filter_distinct_problem_user_id(data_df):
    data_df = filter_distinct_table_key(data_df, 'problem_user_id', max_num=1)
    return data_df


def filter_distinct_problem(data_df, max_num=None):
    data_df = filter_distinct_table_key(data_df, 'problem_id', max_num=max_num)
    return data_df


def filter_distinct_user(data_df, max_num=None):
    data_df = filter_distinct_table_key(data_df, 'user_id', max_num=max_num)
    return data_df


def filter_distinct_test_c_data(data_df):
    data_df = filter_distinct_problem_user_id(data_df)
    data_df = filter_distinct_problem(data_df, 10)
    data_df = filter_distinct_user(data_df, 10)
    return data_df

