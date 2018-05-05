import sys
from types import MethodType

from antlr4 import TokenStream, ParserATNSimulator, ParserRuleContext, CommonTokenStream, Lexer, InputStream
from antlr4.dfa.DFAState import DFAState
from antlr4.error.ErrorListener import ErrorListener
from typing import TextIO

from c_code_processer.c_antlr.CLexer import CLexer
from c_code_processer.c_antlr.CListener import CListener
from c_code_processer.c_antlr.CParser import CParser
from common.constants import CACHE_DATA_PATH
from common.util import disk_cache

is_debug = False
max_token_count = 500
is_monitor = True


@disk_cache(basename='read_parser_train_dfa', directory=CACHE_DATA_PATH)
def read_parser_train_dfa():
    pass


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


def create_monitor_parser(code):
    code_stream = InputStream(code)

    lexer = CLexer(code_stream)
    lexer.addErrorListener(ExceptionErrorListener())

    stream = MonitorTokenStream(lexer)
    # stream = CommonTokenStream(lexer)

    parser = MonitorParser(stream)
    # parser = CParser(stream)
    parser = create_register_enter_listener_parser(parser)
    return code_stream, lexer, stream, parser


def extract_info(parser):
    token = parser.getCurrentToken()
    # print(token)
    state = parser.state
    if is_debug:
        print('in extract info: ', state, token)
    if state != -1:
        expected_toks = parser.getExpectedTokens()
        toks = [t for t in expected_toks]
    else:
        toks = None
    return token, state, toks


# def extrace_token_to_dict(token, symbolicNames):
def extrace_token_to_dict(token):
    token_item = {'token_index': token.tokenIndex, 'value': token.text, 'start': token.start,
                  'stop': token.stop, 'label_id': token.type,
                  # 'stop': token.stop, 'label_id': token.type, 'label': symbolicNames[token.type],
                  'line': token.line, 'column': token.column, 'channel': token.channel}
    return token_item


# ------------------------------------------- parser info record class ----------------------------#
class TokenRecords:

    decisions_to_DFA = read_parser_train_dfa()
    decision_to_state = [int(str(i)) for i in CParser.atn.decisionToState]
    symbolic_ames = CParser.symbolicNames
    # atn state which lookahead one token using LA. total 60
    predict_by_LA_states = [187, 234, 280, 577, 588, 613, 618, 643, 651, 659, 703, 718, 738, 749, 752, 759, 774, 786,
                           799, 803, 817, 826, 829, 832, 839, 841, 850, 859, 914, 918, 941, 944, 950, 966, 980, 983,
                           990, 1009, 1039, 1046, 1073, 1082, 1101, 1104, 1112, 1115, 1119, 1137, 1141, 1160, 1179,
                           1201, 1206, 1210, 1213, 1217, 1221, 1251, 1261, 1285]
    # atn state which using in while loop. the state is also predict but map to decision_to_state.index(state - 2) total 40
    loop_end_predict_states = [226, 297, 308, 366, 380, 394, 414, 428, 439, 450, 461, 472, 483, 512, 552, 585, 610,
                               640, 680, 726, 793, 819, 843, 873, 891, 911, 926, 938, 974, 1017, 1022, 1053, 1066,
                               1103, 1114, 1121, 1154, 1241, 1274, 1298]
    # atn state using recovery error. if it occurs, code is error
    error_recovery_states = [333, 500, 560, 563, 566, 599, 695, 697, 823, 834, 1093, 1094]

    def __init__(self):
        self.total_records = None
        # self.state_records = []
        # self.rule_records = []

        self._tokens = None
        self._tokens_lock = None

        self.last_ctx = None
        self.last_state = -1

        self.in_simulator = False
        self.simulator_item_set = set()

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        self._tokens = value
        self._tokens_lock = [False for i in range(len(value))]
        self.total_records = [None for i in range(len(value))]

    def add_state_record(self, parser, token, before_state, atnState, label_id_list, ctx):
        self.last_state = atnState
        self.last_ctx = ctx
        if label_id_list is None:
            label_id_list = []
        if is_debug:
            self.print_record(parser, token, before_state, atnState, label_id_list)
        if self.check_before_restore(token, atnState):
            self.restore_info(parser, token, before_state, atnState, label_id_list, ctx=ctx, is_state=True)

    def add_rule_record(self, parser, token, state, label_id_list, ctx):
        self.last_state = state
        self.last_ctx = ctx
        if label_id_list is None:
            label_id_list = []
        if is_debug:
            self.print_record(parser, token, None, state, label_id_list)
        if self.check_before_restore(token, state):
            self.restore_info(parser, token, -1, state, label_id_list, ctx=ctx, is_state=False)

    def restore_info(self, parser, token, before_state, atnState, label_id_list, ctx, is_state=True):
        token_item = extrace_token_to_dict(token)
        token_index = token_item['token_index']
        if is_debug:
            print('token index: {}'.format(token_index))

        one = {'token': token_item, 'next_label': label_id_list, 'before_state': before_state, 'state': atnState}
            # before_state, 'state': atnState, 'ctx': ctx}
        self.total_records[token_index] = one

    def print_record(self, parser, token, before_state, atnState, label_id_list):
        print('token info index: {}, start: {}, stop: {}, value: {}, type: {}, label: {}, line: {}, '
              'column: {}'.format(token.tokenIndex, token.start, token.stop, token.text, token.type,
                                  parser.symbolicNames[token.type], token.line, token.column))
        print('original state: {}, state: {}'.format(before_state, atnState))
        if label_id_list is None:
            label_id_list = []
        label_list = [parser.symbolicNames[label_id] for label_id in label_id_list]
        print('next label list: {}'.format(list(zip(label_id_list, label_list))))

    def check_before_restore(self, token, state):
        if not is_monitor:
            return False
        token_index = token.tokenIndex
        if self._tokens_lock[token_index]:
           return False
        decision_state = state - 2 if state in self.loop_end_predict_states else state
        if decision_state in self.predict_by_LA_states:
            self._tokens_lock[token_index] = True
            return True
        return True

    def access_token(self, i):
        if is_monitor and self.in_simulator:
            self.simulator_item_set.add(i)

    def start_simulate(self, ctx):
        self.last_ctx = ctx
        if is_debug:
            print('start simulate')
        self.in_simulator = True
        self.simulator_item_set = set()

    def end_simulate(self):
        if not is_monitor:
            return
        if is_debug:
            print('end simulate: last state: {}'.format(self.last_state))
        self.in_simulator = False
        predict_token_index = sorted(list(self.simulator_item_set))
        start = predict_token_index[0]
        for i in predict_token_index:
            if self._tokens_lock[i]:
                continue
            self._tokens_lock[i] = True
            dfa, s0 = self.deal_with_dfa(self.last_ctx.parser, self.last_state, self.last_ctx)
            expected_tok = self.calculate_expected_token_using_dfa(s0, start, i)
            if len(expected_tok) == 0:
                self._tokens_lock[i] = False
                continue
            token = self.tokens[i]
            self.restore_info(self.last_ctx.parser, token, -1, self.last_state, expected_tok, self.last_ctx)

    def deal_with_dfa(self, parser, state, ctx):
        decision_state = state - 2 if state in self.loop_end_predict_states else state
        decision = self.decision_to_state.index(decision_state)

        dfa = self.decisions_to_DFA[decision]
        s0 = get_s0(dfa, parser._interp, ctx)
        return dfa, s0

    def calculate_expected_token_using_dfa(self, s0, start, i):
        cur = start
        cur_s0 = s0
        while cur < i:
            tok = self.tokens[cur]
            if cur_s0.edges is None or len(cur_s0.edges) < 1 or cur_s0.isAcceptState:
                print('calculate dfa failed. s0 is end points: edges: {}, {}'.format(cur_s0.edges, cur_s0.isAcceptState))
                print('start: {}, i: {}, cur: {}'.format(start, i, cur))
                return None
            next_s = cur_s0.edges[tok.type + 1]
            if next_s is None:
                print('calculate dfa failed. path error')
                return None
            cur_s0 = next_s
            cur += 1

        edges = cur_s0.edges
        if cur_s0.isAcceptState:
            return []
        if edges is None:
            print('edges is None: {}, {}, {}, {}'.format(cur_s0.isAcceptState, start, i, len(self.tokens)))
            return None
        expected_tok = []
        for i, e in enumerate(edges):
            if e is not None:
                expected_tok += [i-1]
        return expected_tok




global_recorder = TokenRecords()
def get_global_recorder():
    return global_recorder

def set_global_recorder(recorder:TokenRecords):
    global global_recorder
    del global_recorder
    global_recorder = recorder


# -------------------------------- state monitor decorator -------------------------------- #
def record_state_wrapper(f):
    def record_state(*args, **kwargs):
        parser = args[0] if len(args) > 0 else kwargs['self']
        atnState = args[1] if len(args) >= 2 else kwargs['atnState']
        before_state = parser.state

        if is_debug:
            print('in monitor all: ', before_state, atnState)
        res = f(*args, **kwargs)
        if is_monitor:
            token, state, toks = extract_info(parser)
            global_recorder.add_state_record(parser, token, before_state, state, toks, parser._ctx)
        if is_debug:
            print('end monitor all: ', before_state, atnState)
        return res
    return record_state


class MonitorParser(CParser):

    decisionsToDFA = read_parser_train_dfa()

    @property
    def state(self):
        return self._stateNumber

    @state.setter
    @record_state_wrapper
    def state(self, atnState: int):
        self._stateNumber = atnState

    def __init__(self, input: TokenStream, output: TextIO = sys.stdout):
        super(MonitorParser, self).__init__(input, output)
        self._interp = MonitorATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)

# ---------------------------------- monitor predict in ParserATNSimulator ------------------------ #

def adaptivePredict_wrapper(f):
    def message_in_predict(*args, **kwargs):
        decision = args[2]
        ctx = args[3]
        state = args[0].parser.atn.decisionToState[decision]
        if is_debug:
            print('start customer adaptivePredict: decision: {}, state: {}'.format(decision, state))
        record = get_global_recorder()
        record.start_simulate(ctx)
        res = f(*args, **kwargs)
        record.end_simulate()
        if is_debug:
            print('end customer adaptivePredict')
        return res
    return message_in_predict


class MonitorATNSimulator(ParserATNSimulator):
    @adaptivePredict_wrapper
    def adaptivePredict(self, input: TokenStream, decision: int, outerContext: ParserRuleContext):
        return super(MonitorATNSimulator, self).adaptivePredict(input, decision, outerContext)


# ----------------------------------- token list access monitor ------------------------ #
class MonitorTokenList(list):
    def __getitem__(self, item):
        if is_monitor:
            if is_debug:
                print('item access : {}'.format(item))
            record = get_global_recorder()
            record.access_token(item)
        res = super(MonitorTokenList, self).__getitem__(item)
        return res


class MonitorTokenStream(CommonTokenStream):
    def __init__(self, lexer:Lexer):
        super(MonitorTokenStream, self).__init__(lexer)
        self.fill()
        self.tokens = MonitorTokenList(self.tokens)


# ------------------------------------------- create parser-sync listener --------------------------------- #
def create_register_enter_listener_parser(parser):
    # parser = register_parser_listener(RecordEnterListener, enter_all_fn, parser)
    parser.addErrorListener(ExceptionErrorListener())
    return parser


def register_parser_listener(cls, enter_fn, parser):
    customer_listener = cls()
    original_keys = cls.__dict__.keys()
    listener_dict = {}
    for key, value in CListener.__dict__.items():
        if key[:5] == 'enter' and key not in original_keys:
            listener_dict[key] = value
    for key in listener_dict.keys():
        customer_listener.__dict__[key] = MethodType(enter_fn, customer_listener)
    parser.addParseListener(customer_listener)
    return parser


def enter_all_fn(self, ctx):
    parser = ctx.parser
    if is_debug:
        print('in enter: ', parser.state)
    # if is_monitor:
    #     token, state, toks = extract_info(parser)
    #     global_recorder.add_rule_record(parser, token, state, toks, ctx)
    if is_debug:
        print('end enter: ', parser.state)


class RecordEnterListener(CListener):
    pass

class ExceptionErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        full_msg = "syntaxError: line " + str(line) + ":" + str(column) + " " + msg
        raise Exception(full_msg)


if __name__ == '__main__':
    s = '''int main(){
return 0;
    }'''
    create_monitor_parser(s)
