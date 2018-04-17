from read_data.read_experiment_data import read_distinct_problem_user_ac_c99_code_dataset, read_filtered_distinct_problem_user_ac_c99_code_dataset
from c_code_processer.code_util import extract_include_from_code

import more_itertools

from collections import Counter


def find_header_in_list(code_list):
    return Counter(more_itertools.flatten(map(extract_include_from_code, code_list)))


if __name__ == '__main__':
    print("load data")
    train_df, valid_df, test_df = read_filtered_distinct_problem_user_ac_c99_code_dataset()
    print("data loaded")
    print("find system header in train_data")
    r = find_header_in_list(train_df['code'])
    print("The code in the train dataset:{}".format(len(train_df['code'])))
    print("The code in the train dataset:{}".format(len(set(train_df['code']))))
    print("find system header in valid_data")
    r += find_header_in_list(valid_df['code'])
    print("The code in the train dataset:{}".format(len(valid_df['code'])))
    print("The code in the train dataset:{}".format(len(set(valid_df['code']))))
    print('find system header in test_data')
    r += find_header_in_list(test_df['code'])
    print("The code in the train dataset:{}".format(len(test_df['code'])))
    print("The code in the train dataset:{}".format(len(set(test_df['code']))))
    print(r)
