import os
from config import root, scrapyOJ_path, cache_path

# scrapyOJ db path. all OJ data
scrapyOJ_DB_PATH = scrapyOJ_path


# project dir path
ROOT_PATH = root
DATA_PATH = os.path.join(ROOT_PATH, 'data')
CACHE_DATA_PATH = cache_path
TMP_FILE_PATH = os.path.join(ROOT_PATH, 'tmp')


# db path
TRAIN_DATA_DBPATH = os.path.join(DATA_PATH, 'train_data.db')
COMPILE_SUCCESS_DATA_DBPATH = os.path.join(DATA_PATH, 'compile_success_data.db')


# table name
ACTUAL_C_ERROR_RECORDS = 'actual_c_error_records'
CPP_TESTCASE_ERROR_RECORDS = 'cpp_testcase_error_records'
C_COMPILE_SUCCESS_RECORDS = 'c_compile_success_records'


# code status and language transform dict
verdict = {'OK': 1, 'REJECTED': 2, 'WRONG_ANSWER': 3, 'RUNTIME_ERROR': 4, 'TIME_LIMIT_EXCEEDED': 5, 'MEMORY_LIMIT_EXCEEDED': 6,
           'COMPILATION_ERROR': 7, 'CHALLENGED': 8, 'FAILED': 9, 'PARTIAL': 10, 'PRESENTATION_ERROR': 11, 'IDLENESS_LIMIT_EXCEEDED': 12,
           'SECURITY_VIOLATED': 13, 'CRASHED': 14, 'INPUT_PREPARATION_CRASHED': 15, 'SKIPPED': 16, 'TESTING': 17, 'SUBMITTED': 18}
langdict = {'GNU C': 1, 'GNU C11': 2, 'GNU C++': 3, 'GNU C++11': 4, 'GNU C++14': 5,
            'MS C++': 6, 'Mono C#': 7, 'MS C#': 8, 'D': 9, 'Go': 10,
            'Haskell': 11, 'Java 8': 12, 'Kotlin': 13, 'Ocaml': 14, 'Delphi': 15,
            'FPC': 16, 'Perl': 17, 'PHP': 18, 'Python 2': 19, 'Python 3': 20,
            'PyPy 2': 21, 'PyPy 3': 22, 'Ruby': 23, 'Rust': 24, 'Scala': 25,
            'JavaScript': 26}