import os
import pandas as pd

from common.analyse_include_util import replace_include_with_blank
from common.constants import ROOT_PATH
from read_data.load_parsed_data import parse_c99_code_to_token

example_code_path=os.path.join(ROOT_PATH, 'example.c')


def read_example_code_tokens():
    code = read_code_from_file()
    code = replace_include_with_blank(code)
    df = pd.DataFrame({'code': [code]})
    res = parse_c99_code_to_token(df)
    return res


def read_code_from_file():
    text = ''
    with open(example_code_path) as f:
        text = f.read()
    return text
