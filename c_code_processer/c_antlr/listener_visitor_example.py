from antlr4 import *
from antlr4.atn.ATNState import ATNState
from antlr4.dfa.DFAState import DFAState

from c_code_processer.antlr_util import MonitorParser, create_register_enter_listener_parser
from c_code_processer.c_antlr.CLexer import CLexer
from c_code_processer.c_antlr.CVisitor import CVisitor




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

    s_stream = InputStream(s)
    lexer = CLexer(s_stream)
    stream = CommonTokenStream(lexer)

    parser = MonitorParser(stream)
    parser = create_register_enter_listener_parser(parser)
    print('decisionToState: ', len(parser.atn.decisionToState))
    dec = []
    for state in parser.atn.decisionToState:
        if state in dec:
            print('multiple state: ', str(state))
        dec += [str(state)]
        print(str(state))

    tree = parser.compilationUnit()
    from c_code_processer.antlr_util import global_recorder
    for record in global_recorder.total_records:
        state_str = str(record['state'])
        if state_str in dec:
            decision = dec.index(state_str)
            print('decision: ', decision)
            dfa = parser.decisionsToDFA[decision]
            # print(parser.decisionsToDFA[decision])
            s0 = get_s0(dfa, parser._interp, record['ctx'])
            if s0.edges is not None:
                print(record['token'])
                for i in record['next_label']:
                    print(i)
                print(len(s0.edges))
                print()
            else:
                print('edges is None')
    # print('final: ')
    # for dfa in parser.decisionsToDFA:
    #     try:
    #         print('one: ', dfa.toString(parser.literalNames))
    #     except Exception as e:
    #         print('pass')


