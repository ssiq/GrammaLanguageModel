import os
import sys

from antlr4 import *
from antlr4.dfa.DFAState import DFAState

from c_code_processer.antlr_util import create_monitor_parser, global_recorder

# class TestVisitor(CVisitor):
#     def visitJumpStatement(self, ctx):
        # pass
        # print(ctx.getRuleIndex())
        # print(ruleNames[ctx.getRuleIndex()])
        # for i in range(ctx.getChildCount()):
        #     token = ctx.children[i].getPayload()
        #     print(token.text)
        #     print(token.tokenIndex)
        #     print(symbolicNames[token.type])

    # def visitFunctionDefinition(self, ctx):
        # print(ctx.getRuleIndex())
        # print(ruleNames[ctx.getRuleIndex()])
        # for i in range(ctx.getChildCount()):
        #     chi = ctx.children[i]
        #     print(type(chi))
        #     print(isinstance(chi, ParserRuleContext))
        #     if isinstance(chi, ParserRuleContext):
        #         print(ctx.getRuleIndex())
        #         print(ruleNames[ctx.getRuleIndex()])
            # token = ctx.children[i].getPayload()
            # print(token.text)
            # print(token.tokenIndex)
            # print(symbolicNames[token.type])
        # return self.visitChildren(ctx)
from c_code_processer.c_antlr.CLexer import CLexer
from c_code_processer.c_antlr.CParser import CParser
from common.constants import CACHE_DATA_PATH
from common.util import disk_cache


def get_s0(dfa, parser_atn_simulator, outerContext):
    if dfa.precedenceDfa:
        # the start state for a precedence DFA depends on the current
        # parser precedence, and is provided by a DFA method.
        s0 = dfa.getPrecedenceStartState(parser_atn_simulator.parser.getPrecedence())
    else:
        # the start state for a "regular" DFA is just s0
        s0 = dfa.s0

    if s0 is None:
        if outerContext is None:
            outerContext = ParserRuleContext.EMPTY
        if ParserATNSimulator.debug or ParserATNSimulator.debug_list_atn_decisions:
            print("predictATN decision " + str(dfa.decision) +
                  " exec LA(1)==" + parser_atn_simulator.getLookaheadName(input) +
                  ", outerContext=" + outerContext.toString(parser_atn_simulator.parser.literalNames, None))

        fullCtx = False
        s0_closure = parser_atn_simulator.computeStartState(dfa.atnStartState, ParserRuleContext.EMPTY, fullCtx)

        if dfa.precedenceDfa:
            # If this is a precedence DFA, we use applyPrecedenceFilter
            # to convert the computed start state to a precedence start
            # state. We then use DFA.setPrecedenceStartState to set the
            # appropriate start state for the precedence level rather
            # than simply setting DFA.s0.
            #
            dfa.s0.configs = s0_closure  # not used for prediction but useful to know start configs anyway
            s0_closure = parser_atn_simulator.applyPrecedenceFilter(s0_closure)
            s0 = parser_atn_simulator.addDFAState(dfa, DFAState(configs=s0_closure))
            dfa.setPrecedenceStartState(parser_atn_simulator.parser.getPrecedence(), s0)
        else:
            s0 = parser_atn_simulator.addDFAState(dfa, DFAState(configs=s0_closure))
            dfa.s0 = s0
    return s0


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    s = '''int f(int arg1, char arg2)
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

    s2 = r'''int main(void)
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
}'''

    code = s2

    # code_stream = InputStream(code)
    # lexer = CLexer(code_stream)
    # stream = CommonTokenStream(lexer)
    # parser = CParser(stream)

    _, _, stream, parser = create_monitor_parser(code)
    global_recorder.tokens = stream.tokens
    # print('decisionToState: ', len(parser.atn.decisionToState))
    # dec = []
    # for i, state in enumerate(parser.atn.decisionToState):
    #     if state in dec:
    #         print('multiple state: ', str(state))
    #     dec += [str(state)]
    #     print(i, str(state))

    # parser2 = MonitorParser(stream)
    # parser2 = create_register_enter_listener_parser(parser2)
    # tree2 = parser2.compilationUnit()
    # print(parser._interp, parser2._interp)
    # for dfa1, dfa2 in zip(parser.decisionsToDFA, parser2.decisionsToDFA):
    #     print('DFA compare: ', id(dfa1), id(dfa2), id(dfa1) == id(dfa2))
    #     if id(dfa1) != id(dfa2):
    #         print('DFA different!!!')



    # @disk_cache(basename='read_antlr_parse_records', directory=os.path.join(CACHE_DATA_PATH))
    # def get_dfa():
    #     print('in get dfa')
    #     return parser.decisionsToDFA


    # res = get_dfa()
    # parser.decisionsToDFA = res

    tree = parser.compilationUnit()


    @disk_cache(basename='test', directory=CACHE_DATA_PATH)
    def test():
        return parser._predicates

    pre = parser._predicates

    te = test()
    record = global_recorder
    pass
    # from c_code_processer.antlr_util import global_recorder
    # for record in global_recorder.total_records:
    #     state_str = str(record['state'])
    #     if state_str in dec:
    #         decision = dec.index(state_str)
    #         print('state: {}, decision: {}'.format(state_str, decision))
    #         print('decision: ', decision)
    #         dfa = parser.decisionsToDFA[decision]
    #         # print(parser.decisionsToDFA[decision])
    #         s0 = get_s0(dfa, parser._interp, record['ctx'])
    #         if s0.edges is not None:
    #             print(record['token'])
    #             for i in record['next_label']:
    #                 print(i)
    #             print(len(s0.edges))
    #             print()
    #         else:
    #             print('edges is None')

    # print('final: ')
    # for dfa in parser.decisionsToDFA:
    #     try:
    #         print('one: ', dfa.toString(parser.literalNames))
    #     except Exception as e:
    #         print('pass')


