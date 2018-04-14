from common.util import disk_cache
from common.constants import CACHE_DATA_PATH
from c_code_processer.code_util import tokenize
from read_data.read_experiment_data import read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset

import more_itertools

@disk_cache(basename='read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset', directory=CACHE_DATA_PATH)
def load_tokenized_filtered_without_include_distinct_problem_user_ac_c99_code_dataset():
    def tokenize_df(df):
        def f(code):
            tokens = tokenize(code)
            return [token.value for token in tokens]
        codes = map(f, df['code'])
        return list(codes)
    return [tokenize_df(df) for df in read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()]


@disk_cache(basename='get_token_vocabulary', directory=CACHE_DATA_PATH)
def get_token_vocabulary():
    train, _, _ = load_tokenized_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()
    return set(more_itertools.flatten(train))





