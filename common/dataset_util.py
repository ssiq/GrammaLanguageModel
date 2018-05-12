from pycparser.ply.lex import LexToken

from common.action_constants import ActionType
from common.analyse_include_util import extract_include, replace_include_with_blank
from common.generate_token_util import generate_mark_token_action
from common.parse_util import tokenize_by_clex_fn
from read_data.read_experiment_data import read_fake_common_c_error_dataset, read_fake_random_c_error_dataset

import json


def convert_one_operation_dict_to_list(operation):
    act_type = ActionType(operation['act_type'])
    token_pos = operation['token_pos']
    token_text = operation['to_char']
    return [act_type, token_pos, token_text]


def create_error_tokens_by_operations(one):
    ac_tokens = one['ac_tokens']
    operations = json.loads(one['modify_action_list'])

    operation_list = [convert_one_operation_dict_to_list(ope) for ope in operations]

    try:
        tokens, _ = generate_mark_token_action(operation_list, ac_tokens)
    except Exception as e:
        tokens = None
    # print(len(ac_tokens), len(tokens))
    # for i, token in enumerate(tokens):
    #     print(i, ac_tokens[i], token)
    one['tokens'] = tokens
    return one


def create_error_tokens(df, max_length=498):
    df['res'] = ''
    df['includes'] = df['similar_code'].map(extract_include)
    df['similar_code_without_includes'] = df['similar_code'].map(replace_include_with_blank)
    tokenize_fn = tokenize_by_clex_fn()
    df['ac_tokens'] = df['similar_code_without_includes'].map(tokenize_fn)
    df = df[df['ac_tokens'].map(lambda x: x is not None and len(x) < max_length)]

    df = df.apply(create_error_tokens_by_operations, raw=True, axis=1)
    df = df[df['tokens'].map(lambda x: x is not None and len(x) < max_length)]

    return df


if __name__ == '__main__':
    train_df, valid_df, test_df = read_fake_random_c_error_dataset()
    print('train: {}, valid: {}, test: {}'.format(len(train_df), len(valid_df), len(test_df)))

    # create_error_tokens_by_operations(tmp_init_one(train_df.iloc[0]))
    # df = read_fake_common_c_error_dataset()

