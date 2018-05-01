from c_code_processer.antlr_util import MonitorParser, create_register_enter_listener_parser, \
    TokenRecords, extrace_token_to_dict, set_global_recorder, get_global_recorder, ExceptionErrorListener
from c_code_processer.c_antlr.CLexer import CLexer
from c_code_processer.code_util import replace_include_with_blank, replace_define_with_blank

from antlr4 import InputStream, CommonTokenStream
import sys


max_token_count = 500

def collect_antlr_parser(code):
    global_recorder = TokenRecords()
    set_global_recorder(global_recorder)

    code_stream = InputStream(code)
    lexer = CLexer(code_stream)
    lexer.addErrorListener(ExceptionErrorListener())
    stream = CommonTokenStream(lexer)
    parser = MonitorParser(stream)
    stream.fill()
    tokens = stream.tokens
    if len(tokens) >= max_token_count:
        return None, None
    tokens = [extrace_token_to_dict(tok, parser.symbolicNames) for tok in tokens]

    parser = create_register_enter_listener_parser(parser)

    tree = parser.compilationUnit()
    global_recorder = get_global_recorder()
    return tokens, global_recorder

count = 0
failed = 0
def collect_one_records(one, total):
    global count, failed
    count += 1
    print('in one {}/{}/{}'.format(count, failed, total))
    code = one['code']
    if 'include' in code:
        code = replace_include_with_blank(code)
    if 'define' in code:
        code = replace_define_with_blank(code)
    try:
        tokens, records = collect_antlr_parser(code)
    except Exception as e:
        print('failed occurs')
        print(e)
        failed += 1
        tokens = None
        records = None

    one['tokens'] = tokens
    one['parse_records'] = records
    if tokens is None:
        return one
    print(len(tokens))
    print(len(records.total_records), len(records.state_records), len(records.rule_records))
    sys.stdout.flush()
    sys.stderr.flush()
    return one


if __name__ == '__main__':

    s = ''' #include <stdio.h>
    int f(int arg1, char arg2)
{
	a1(arg1);
	a2(arg1, arg2);
	a3();
}

int f2(int arg1, char arg2)
{
	a1(arg1);
	a2(arg1, arg2);
	a3();
	while(True){
	    break;
	}
}
'''

    tokens, global_recorder = collect_antlr_parser(s)
    # print(len(tokens))
    # print(len(global_recorder.total_records), len(global_recorder.state_records), len(global_recorder.rule_records))
