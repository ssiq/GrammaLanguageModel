from read_data.read_filter_data_records import read_distinct_problem_user_compile_success_c_records, \
    read_distinct_problem_user_fake_c_random_records, read_distinct_problem_user_fake_c_common_records, \
    read_distinct_problem_user_c_records
from common.util import disk_cache
from common.constants import CACHE_DATA_PATH
from c_code_processer.code_util import extract_include_from_code, remove_include


def filter_frac(data_df, frac):
    user_count = len(data_df.groupby('user_id').size())
    print('user_count: {}'.format(user_count))
    user_id_list = data_df['user_id'].sample(int(user_count * frac)).tolist()
    print('user_id_list: {}'.format(len(user_id_list)))
    split_df = data_df[data_df.apply(lambda x: x['user_id'] in user_id_list, axis=1, raw=True)]
    print('split_df: {}'.format(len(split_df)))
    # drop_df = data_df[data_df.apply(lambda x: x['user_id'] in user_id_list, axis=1, raw=True)]
    main_df = data_df.drop(split_df.index)
    print('main_df: {}'.format(len(main_df)))
    return main_df, split_df


@disk_cache(basename='read_distinct_problem_user_ac_c99_code_dataset', directory=CACHE_DATA_PATH)
def read_distinct_problem_user_ac_c99_code_dataset():
    data_df = read_distinct_problem_user_compile_success_c_records()

    main_df, test_df = filter_frac(data_df, 0.1)
    train_df, vaild_df = filter_frac(main_df, 0.1)
    print('train df size: {}, valid df size: {}, test df size: {}'.format(len(train_df), len(vaild_df), len(test_df)))
    return train_df, vaild_df, test_df


@disk_cache(basename='read_filtered_distinct_problem_user_ac_c99_code_dataset', directory=CACHE_DATA_PATH)
def read_filtered_distinct_problem_user_ac_c99_code_dataset():
    maintain_header_file = {'stdio.h', 'stdlib.h', 'string.h', 'math.h'}

    def filter_header(df):

        def not_contain_other_header(code):
            for t in extract_include_from_code(code):
                if t not in maintain_header_file:
                    return False
            return True

        return df[df['code'].map(not_contain_other_header)]
    return [filter_header(df) for df in read_distinct_problem_user_ac_c99_code_dataset()]


@disk_cache(basename='read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset', directory=CACHE_DATA_PATH)
def read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset():
    def remove_include_df(df):
        df['code'] = df['code'].map(remove_include)
        return df

    return [remove_include_df(df) for df in read_filtered_distinct_problem_user_ac_c99_code_dataset()]


@disk_cache(basename='read_fake_common_c_error_dataset', directory=CACHE_DATA_PATH)
def read_fake_common_c_error_dataset():
    test_df = read_distinct_problem_user_c_records()
    test_df = test_df[test_df['distance'].map(lambda x: 0 < x < 10)]
    data_df = read_distinct_problem_user_fake_c_common_records()
    train_df, valid_df = filter_frac(data_df, 0.1)
    print('train df size: {}, valid df size: {}, test df size: {}'.format(len(train_df), len(valid_df), len(test_df)))
    return train_df, valid_df, test_df


@disk_cache(basename='read_fake_random_c_error_dataset', directory=CACHE_DATA_PATH)
def read_fake_random_c_error_dataset():
    test_df = read_distinct_problem_user_c_records()
    test_df = test_df[test_df['distance'].map(lambda x: 0 < x < 10)]
    data_df = read_distinct_problem_user_fake_c_random_records()
    train_df, valid_df = filter_frac(data_df, 0.1)
    print('train df size: {}, valid df size: {}, test df size: {}'.format(len(train_df), len(valid_df), len(test_df)))
    return train_df, valid_df, test_df


if __name__ == '__main__':
    # data_df = read_distinct_problem_user_compile_success_c_records()
    # print('all data df', len(data_df))
    # main_df, split_df = filter_frac(data_df, 0.1)
    # print('train_df length: {}, split_df length: {}'.format(len(main_df), len(split_df)))
    read_distinct_problem_user_ac_c99_code_dataset()
    read_filtered_distinct_problem_user_ac_c99_code_dataset()
    read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()
