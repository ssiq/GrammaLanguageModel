import errno
import functools
import itertools
import os
import pickle
import types
from multiprocessing import Pool
import typing
import hashlib

import more_itertools
import sklearn
import pandas as pd
import sys


def make_dir(*path: str) -> None:
    """
    This method will recursively create the directory
    :param path: a variable length parameter
    :return:
    """
    path = os.path.join(*path)

    if not path:
        return

    if os.path.exists(path):
        if not os.path.isdir(path):
            raise ValueError("The path {} already exits but it is not a directory".format(path))
        return

    base, name = os.path.split(path)
    make_dir(base)
    if name:
        os.mkdir(path)


def format_dict_to_string(to_format_dict: dict) -> str:
    """
    :param to_format_dict: a dict to format
    :return:
    """

    def to_str(o):
        if is_sequence(o):
            return ''.join(to_str(t) for t in o)
        else:
            return str(o)
    # print(len('__'.join(to_str(a)+to_str(b) for a, b in to_format_dict.items())))
    return '__'.join(to_str(a)+to_str(b) for a, b in to_format_dict.items())


def ensure_directory(directory):
    """
    Create the directories along the provided directory path that do not exist.
    """
    directory = os.path.expanduser(directory)
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

def disk_cache(basename, directory, method=False):
    """
    Function decorator for caching pickleable return values on disk. Uses a
    hash computed from the function arguments for invalidation. If 'method',
    skip the first argument, usually being self or cls. The cache filepath is
    'directory/basename-hash.pickle'.
    """
    directory = os.path.expanduser(directory)
    ensure_directory(directory)

    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            key = (tuple(args), tuple(kwargs.items()))
            # Don't use self or cls for the invalidation hash.
            if method and key:
                key = key[1:]
            filename = '{}-{}.pickle'.format(basename, data_hash(key))
            print("the cache name is {}".format(filename))
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                print("load file from:{}".format(filepath))
                with open(filepath, 'rb') as handle:
                    return pickle.load(handle)
            result = func(*args, **kwargs)
            with open(filepath, 'wb') as handle:
                print("write cache to: {}".format(filepath))
                pickle.dump(result, handle)
            return result
        return wrapped

    return wrapper


def data_hash(key):

    def hash_value(hash_item):
        v = 0
        try:
            v = int(hashlib.md5(str(hash_item).encode('utf-8')).hexdigest(), 16)
        except Exception as e:
            print('error occur while hash item {} '.format(type(hash_item)))
        return v

    hash_val = 0
    key = list(more_itertools.flatten(key))
    for item in key:
        if isinstance(item, pd.DataFrame):
            serlist = [item.itertuples(index=False, name=None)]
            serlist = list(more_itertools.collapse(serlist))
            for ser in serlist:
                val = hash_value(ser)
                hash_val += val
        elif isinstance(item, pd.Series):
            serlist = item.tolist()
            serlist = list(more_itertools.collapse(serlist))
            for ser in serlist:
                val = hash_value(ser)
                hash_val += val
        elif isinstance(item, int) or isinstance(item, float) or isinstance(item, str):
            val = hash_value(item)
            hash_val += val
        elif isinstance(item, list) or isinstance(item, set) or isinstance(item, tuple):
            serlist = list(more_itertools.collapse(item))
            for ser in serlist:
                val = hash_value(ser)
                hash_val += val
        elif isinstance(item, dict):
            serlist = list(more_itertools.collapse(item.items()))
            for ser in serlist:
                val = hash_value(ser)
                hash_val += val
        else:
            print('type {} cant be hashed.'.format(type(item)))
    return str(hash_val)

# ================================================================
# multiprocess function
# ================================================================

def parallel_map(core_num, f, args):
    """
    :param core_num: the cpu number
    :param f: the function to parallel to do
    :param args: the input args
    :return:
    """

    with Pool(core_num) as p:
        r = p.map(f, args)
        return r

# ================================================================
# dict function
# ================================================================

def reverse_dict(d: dict) -> dict:
    """
    swap key and value of a dict
    dict(key->value) => dict(value->key)
    """
    return dict(map(reversed, d.items()))

# ================================================================
# sequence function
# ================================================================

def is_sequence(s):
    try:
        iterator = iter(s)
    except TypeError:
        return False
    else:
        if isinstance(s, str):
            return False
        return True


def convert_to_list(s):
    if is_sequence(s):
        return list(s)
    else:
        return [s]


def sequence_sum(itr):
    return sum(itr)

def padded_code_new(batch_code):
    if not isinstance(batch_code, list):
        return batch_code
    elif not isinstance(batch_code[0], list):
        return batch_code

    batch_root = batch_code
    while True:
        if not isinstance(batch_root, list):
            return batch_code
        elif not isinstance(batch_root[0], list):
            return batch_code
        fill_value = 0
        if isinstance(batch_root[0][0], list):
            fill_value = []
        max_len = max(map(len, batch_root))
        for b in batch_root:
            while len(b) < max_len:
                b.append(fill_value)
        # list(map(lambda x: list(more_itertools.padded(x, fillvalue=fill_value, n=max_len)), batch_root))

        tmp = []
        for son in batch_root:
            for s in son:
                tmp.append(s)
        batch_root = tmp

def padded(x, deepcopy=False):
    import copy
    if deepcopy:
        x = copy.deepcopy(x)
    if not isinstance(x, list):
        return x
    elif isinstance(x[0], list):
        return padded_code_new(x)
    else:
        return x

def batch_holder(*data: typing.List, batch_size=32,):
    """
    :param data:
    :return:
    """
    def iterator():
        def one_epoch():
            i_data = list(map(lambda x: more_itertools.chunked(x, batch_size), data))
            return zip(*i_data)
        for i ,m in enumerate(more_itertools.repeatfunc(one_epoch, times=1)):
            for t in m:
                yield t

    return iterator

def dataset_holder(*args):
    def f():
        return args
    return f

def train_test_split(data, test_size):
    from sklearn.model_selection import train_test_split
    data = train_test_split(*data, test_size=test_size)

    d_len = len(data)
    train_data = [data[i] for i in range(0, d_len, 2)]
    test_data = [data[i] for i in range(1, d_len, 2)]
    return train_data, test_data

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def get_count_size(x):
    r = 0
    if hasattr(x, '__iter__') and not isinstance(x, (str, bytes, bytearray)):
        r += sum(get_count_size(t) for t in x)
    else:
        r += 1

    return r


def unique_adjacent(seq: typing.Iterator):
    pre = next(seq)
    yield pre
    for token in seq:
        if token == pre:
            continue
        else:
            pre = token
            yield pre


def group_df_to_grouped_list(data_df, groupby_key):
    grouped = data_df.groupby(groupby_key)
    group_list = []
    for name, group in grouped:
        group_list += [group]
    return group_list

def maintain_function_co_firstlineno(ori_fn):
    """
    This decorator is used to make the decorated function's co_firstlineno the same as the ori_fn
    """

    def wrapper(fn):
        wrapper_code = fn.__code__
        fn.__code__ = types.CodeType(
            wrapper_code.co_argcount,
            wrapper_code.co_kwonlyargcount,
            wrapper_code.co_nlocals,
            wrapper_code.co_stacksize,
            wrapper_code.co_flags,
            wrapper_code.co_code,
            wrapper_code.co_consts,
            wrapper_code.co_names,
            wrapper_code.co_varnames,
            wrapper_code.co_filename,
            wrapper_code.co_name,
            ori_fn.__code__.co_firstlineno,
            wrapper_code.co_lnotab,
            wrapper_code.co_freevars,
            wrapper_code.co_cellvars
        )

        return fn

    return wrapper


if __name__ == '__main__':
    make_dir('data', 'cache_data')
