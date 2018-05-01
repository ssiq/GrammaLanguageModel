from antlr4 import *

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

    tree = parser.compilationUnit()


