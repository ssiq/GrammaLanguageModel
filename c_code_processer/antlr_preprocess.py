import multiprocessing

from c_code_processer.antlr_util import TokenRecords, extrace_token_to_dict, set_global_recorder, get_global_recorder, \
    create_monitor_parser
from c_code_processer.c_antlr.CLexer import CLexer
from c_code_processer.code_util import replace_include_with_blank, replace_define_with_blank

from antlr4 import InputStream, CommonTokenStream
import sys


max_token_count = 500
count = 0
failed = 0


def init_parser_and_token(code):
    _, _, stream, parser = create_monitor_parser(code)
    tokens = stream.tokens
    if len(tokens) >= max_token_count:
        return None, None
    # tokens = [extrace_token_to_dict(tok) for tok in tokens]
    return parser, tokens


def collect_antlr_parser(code):
    global_recorder = TokenRecords()
    set_global_recorder(global_recorder)

    parser, tokens = init_parser_and_token(code)
    if tokens is None:
        return None, None
    global_recorder.tokens = tokens
    parser.compilationUnit()
    tokens = [extrace_token_to_dict(tok) for tok in tokens]

    # del code_stream, lexer, stream, parser
    global_recorder = get_global_recorder()
    return tokens, global_recorder.total_records


def collect_dfa_do_parse(code, total):
    if 'include' in code:
        code = replace_include_with_blank(code)
    if 'define' in code:
        code = replace_define_with_blank(code)
    global count, failed
    count += 1
    print('in one code {}/{}/{}'.format(count, failed, total))
    try:
        parser, tokens = init_parser_and_token(code)
        parser.compilationUnit()
    except Exception as e:
        print('failed occured!!!')
        print(e)
        failed += 1
        tokens = None
    return tokens


def collect_one_records(one, total):
    current = multiprocessing.current_process()
    global count, failed
    count += 1
    print('{}, {} in one {}/{}/{}'.format(current.pid, current.name, count, failed, total))
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
        print(code)
        failed += 1
        tokens = None
        records = None

    one['tokens'] = tokens
    one['parse_records'] = records
    if tokens is None:
        return one
    print(len(tokens))
    # print(len(records.total_records), len(records.state_records), len(records.rule_records))
    print(len(records))
    sys.stdout.flush()
    sys.stderr.flush()
    return one


def main():
    s = r'''
int main(void)
{
 int i,j,n;
 double *a,s=0; 
 scanf( "%d" ,&n);
 a=malloc(n*n*sizeof(double));
 for(i=0;i<n;i++)
 {
 for(j=0;j<n;j++)
 {
 scanf( "%lg" ,&a[i*n+j]);
 }
 }
 for(i=0;i<n;i++)
 {
 s+=a[i*n+i];
 s+=a[i*n+n-1-i];
 s+=a[n*(n-1)/2+i];
 s+=a[n*i+(n-1)/2];
 }
 printf( "%lg" ,s-3*a[n*(n-1)/2+(n-1)/2]);
 return 0; 
}
    '''
    # for i in range(100):
    #     tokens, total = collect_antlr_parser(s)
    tokens, total = collect_antlr_parser(s)
    pass
    # print(len(tokens))
    # print(len(total))

if __name__ == '__main__':

    main()
    # print(len(tokens))
    # print(len(global_recorder.total_records), len(global_recorder.state_records), len(global_recorder.rule_records))
