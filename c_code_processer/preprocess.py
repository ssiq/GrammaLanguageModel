import config
import typing
from subprocess import Popen, PIPE


def preprocess_file(filename:str, cpp_path: str='cpp', cpp_args: typing.Union[str, typing.List[str]]='') -> str:
    """ Preprocess a file using cpp.
        filename:
            Name of the file you want to preprocess.
        cpp_path:
        cpp_args:
            Refer to the documentation of parse_file for the meaning of these
            arguments.
        When successful, returns the preprocessed file's contents.
        Errors from cpp will be printed out.
    """
    path_list = [cpp_path]
    if isinstance(cpp_args, list):
        path_list += cpp_args
    elif cpp_args != '':
        path_list += [cpp_args]
    path_list += [filename]

    try:
        # Note the use of universal_newlines to treat all newlines
        # as \n for Python's purpose
        #
        pipe = Popen(   path_list,
                        stdout=PIPE,
                        universal_newlines=True)
        text = pipe.communicate()[0]
    except OSError as e:
        raise RuntimeError("Unable to invoke 'cpp'.  " +
            'Make sure its path was passed correctly\n' +
            ('Original error: %s' % e))

    return text


def preprocess(file_path: str, user_header_paths: typing.List[str]=list()) -> str:
    cpp_args = ['-I{}'.format(config.fake_system_header_path)]
    for s in user_header_paths:
        cpp_args.append('-I{}'.format(s))
    print()
    print(cpp_args)
    return preprocess_file(file_path, cpp_args=cpp_args)