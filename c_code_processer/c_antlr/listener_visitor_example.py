from antlr4 import *
from c_code_processer.c_antlr.CLexer import CLexer
from c_code_processer.c_antlr.CParser import CParser
from c_code_processer.c_antlr.CListener import CListener
from c_code_processer.c_antlr.CVisitor import CVisitor


class TestListener(CListener):
    def enterFunctionDefinition(self, ctx):
        print("Oh, a function definition!")


class TestVisitor(CVisitor):
    def visitFunctionDefinition(self, ctx):
        print("Oh, visit a function definition!")
        return self.visitChildren(ctx)


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
}
'''

    s_stream = InputStream(s)
    lexer = CLexer(s_stream)
    stream = CommonTokenStream(lexer)
    parser = CParser(stream)
    # compilationUnit is the root
    tree = parser.compilationUnit()

    testListener = TestListener()
    testVisitor = TestVisitor()
    walker = ParseTreeWalker()
    walker.walk(testListener, tree)
    testVisitor.visit(tree)
