from c_code_processer.antlr_preprocess import collect_one_records
from common.constants import CACHE_DATA_PATH
from common.util import disk_cache
from read_data.read_experiment_data import read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset


@disk_cache(basename='read_antlr_parse_records_dataset', directory=CACHE_DATA_PATH)
def read_antlr_parse_records_dataset():
    datasets = read_filtered_without_include_distinct_problem_user_ac_c99_code_dataset()
    total = 0
    for df in datasets:
        total += len(df)

    datasets = [df.apply(collect_one_records, raw=True, axis=1, total=total) for df in datasets]
    datasets = [df[df['tokens'].map(lambda x: x is not None)] for df in datasets]
    print('train: {}, valid: {}, test: {}'.format(len(datasets[0]), len(datasets[1]), len(datasets[2])))
    return datasets


if __name__ == '__main__':
    datasets = read_antlr_parse_records_dataset()