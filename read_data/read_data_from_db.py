import pandas as pd
import sqlite3

from common.util import disk_cache
from common.constants import CACHE_DATA_PATH, COMPILE_SUCCESS_DATA_DBPATH, C_COMPILE_SUCCESS_RECORDS, langdict, verdict, \
    COMMON_C_ERROR_RECORDS, FAKE_C_COMPILE_ERROR_DATA_DBPATH, RANDOM_C_ERROR_RECORDS


def merge_and_deal_submit_table(problems_df, submit_df):
    submit_joined_df = submit_df.join(problems_df.set_index('problem_name'), on='problem_name')
    submit_joined_df['time'] = submit_joined_df['time'].str.replace('ms', '').astype('int')
    submit_joined_df['memory'] = submit_joined_df['memory'].str.replace('KB', '').astype('int')
    submit_joined_df['submit_time'] = pd.to_datetime(submit_joined_df['submit_time'])
    submit_joined_df['tags'] = submit_joined_df['tags'].str.split(':')
    submit_joined_df['code'] = submit_joined_df['code'].str.slice(1, -1)
    submit_joined_df['language'] = submit_joined_df['language'].replace(langdict)
    submit_joined_df['status'] = submit_joined_df['status'].replace(verdict)
    return submit_joined_df


def read_data(conn, table, condition=None):
    extra_filter = ''
    if condition is not None:
        extra_filter += ' where '
        condition_str = ['{}{}{}'.format(con[0], con[1], con[2]) for con in condition]
        extra_filter += (' and '.join(condition_str))
    sql = 'select * from {} {}'.format(table, extra_filter)
    data_df = pd.read_sql(sql, conn)
    print('read data sql statment: {}. length:{}'.format(sql, len(data_df.index)))
    return data_df


@disk_cache(basename='read_compile_success_c_records', directory=CACHE_DATA_PATH)
def read_compile_success_c_records():
    conn = sqlite3.connect("file:{}?mode=ro".format(COMPILE_SUCCESS_DATA_DBPATH), uri=True)
    data_df = read_data(conn, C_COMPILE_SUCCESS_RECORDS)
    return data_df


@disk_cache(basename='read_fake_common_c_error_records', directory=CACHE_DATA_PATH)
def read_fake_common_c_error_records():
    conn = sqlite3.connect('file:{}?mode=ro'.format(FAKE_C_COMPILE_ERROR_DATA_DBPATH), uri=True)
    data_df = read_data(conn, COMMON_C_ERROR_RECORDS)
    return data_df


@disk_cache(basename='read_fake_random_c_error_records', directory=CACHE_DATA_PATH)
def read_fake_random_c_error_records():
    conn = sqlite3.connect('file:{}?mode=ro'.format(FAKE_C_COMPILE_ERROR_DATA_DBPATH), uri=True)
    data_df = read_data(conn, RANDOM_C_ERROR_RECORDS)
    return data_df
