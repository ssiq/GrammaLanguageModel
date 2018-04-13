import tensorflow as tf


class GrammarBasedModelOp(object):
    def __init__(self,
                 input_placeholders,
                 grammar_productions):
        self.input_placeholders = input_placeholders
        self.grammar_productions = grammar_productions
        

