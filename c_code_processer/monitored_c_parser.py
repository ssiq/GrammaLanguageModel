from c_code_processer.pycparser.pycparser.c_parser import CParser
from c_code_processer.code_util import create_is_p_fn, get_all_c99_production_vocabulary
from c_code_processer.buffered_clex import BufferedCLex
import inspect
import types


class MonitoredParser(object):
    def __init__(self,
                 lex_optimize=True,
                 lexer=BufferedCLex,
                 lextab='pycparser.lextab',
                 yacc_optimize=True,
                 yacctab='pycparser.yacctab',
                 yacc_debug=False,
                 taboutputdir='',
                 ):
        self._lex_optimize = lex_optimize
        self._lexer = lexer
        self._lextab = lextab
        self._yacc_optimize = yacc_optimize
        self._yacctab = yacctab
        self._yacc_debug = yacc_debug
        self._taboutputdir = taboutputdir
        self._init_production_list()
        self._parser = self._new_parser()

    def _init_production_list(self):
        self._production_list = []
        self._matched_ast_node = []

    def _new_parser(self):
        parser = CParser()
        is_parse_fn = create_is_p_fn()
        parse_fn_tuple_list = filter(lambda x: is_parse_fn(x[0]) and x[0] != "p_error", inspect.getmembers(parser))
        production_vocabulary = get_all_c99_production_vocabulary()

        def patch_fn(fn, doc, name, production):
            def wrapper(parse_self, p):
                self._production_list.append(production)
                res = fn(p)
                self._matched_ast_node.append(res)
                return res

            wrapper.__name__ = name
            wrapper.__doc__ = doc
            return wrapper

        for k, v in parse_fn_tuple_list:
            # print("{}:{}".format(k, v))
            productions = production_vocabulary.get_production_by_production_string(v.__doc__)
            for i, production in enumerate(productions):
                name = k if i == 0 else k+str(i)
                new_method = types.MethodType(patch_fn(v, str(production), name, production), parser)
                setattr(parser, name, new_method)
        parser.build(
            self._lex_optimize,
            self._lexer,
            self._lextab,
            self._yacc_optimize,
            self._yacctab,
            self._yacc_debug
        )
        return parser

    def parse(self, text, filename='', debuglevel=0):
        """
        :param text: the code string
        :return: the ast
        """
        self._init_production_list()
        return self._parser.parse(text, filename, debuglevel)

    def parse_get_production_list_and_token_list(self, code):
        final_ast = self.parse(code)
        tokens = [t[0] for t in self._parser.clex.tokens_buffer]
        return self._production_list, self._matched_ast_node, tokens

    def __getattr__(self, item):
        return getattr(self._parser, item)


if __name__ == '__main__':
    monitor = MonitoredParser(lex_optimize=False,
                yacc_debug=True,
                yacc_optimize=False,
                yacctab='yacctab')
    code = """
    int add(int a, int b)
    {
        return a+b;
    }
    """
    print(monitor.parse(code))
    print(monitor.parse_get_production_list_and_token_list(code))

