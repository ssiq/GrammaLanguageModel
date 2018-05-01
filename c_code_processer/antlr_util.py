from types import MethodType

from antlr4.error.ErrorListener import ErrorListener

from c_code_processer.c_antlr.CListener import CListener
from c_code_processer.c_antlr.CParser import CParser


is_debug = False

def extract_info(parser):
    token = parser.getCurrentToken()
    # print(token)
    state = parser.state
    if is_debug:
        print('in extract info: ', state, token)
    if state != -1:
        toks = parser.getExpectedTokens()
    else:
        toks = None
    return token, state, toks


def extrace_token_to_dict(token, symbolicNames):
    token_item = {'tokenIndex': token.tokenIndex, 'value': token.text, 'start': token.start,
                  'stop': token.stop, 'label_id': token.type, 'label': symbolicNames[token.type],
                  'line': token.line, 'column': token.column, 'channel': token.channel}
    return token_item


# ------------------------------------------- parser info record class ----------------------------#
class TokenRecords:

    def __init__(self):
        self.total_records = []
        self.state_records = []
        self.rule_records = []

    def add_state_record(self, parser, token, before_state, atnState, label_id_list):
        if label_id_list is None:
            label_id_list = []
        label_list = [parser.symbolicNames[label_id] for label_id in label_id_list]
        if is_debug:
            self.print_record(parser, token, before_state, atnState, label_id_list, label_list)
        self.restore_info(parser, token, before_state, atnState, label_id_list, label_list, is_state=True)

    def add_rule_record(self, parser, token, state, label_id_list):
        if label_id_list is None:
            label_id_list = []
        label_list = [parser.symbolicNames[label_id] for label_id in label_id_list]
        if is_debug:
            self.print_record(parser, token, None, state, label_id_list, label_list)
        self.restore_info(parser, token, -1, state, label_id_list, label_list, is_state=False)

    def restore_info(self, parser, token, before_state, atnState, label_id_list, label_list, is_state=True):
        token_item = extrace_token_to_dict(token, parser.symbolicNames)

        labels = list(zip(label_id_list, label_list))

        one = {'is_state': is_state, 'token': token_item, 'next_label': labels, 'before_state':
            before_state, 'state': atnState}
        self.total_records += [one]
        if is_state:
            self.state_records += [one]
        else:
            self.rule_records += [one]

    def print_record(self, parser, token, before_state, atnState, label_id_list, label_list):
        print('token info index: {}, start: {}, stop: {}, value: {}, type: {}, label: {}, line: {}, '
              'column: {}'.format(token.tokenIndex, token.start, token.stop, token.text, token.type,
                                  parser.symbolicNames[token.type], token.line, token.column))
        print('original state: {}, state: {}'.format(before_state, atnState))
        if label_id_list is None:
            label_id_list = []
        print('next label list: {}'.format(list(zip(label_id_list, label_list))))


global_recorder = TokenRecords()
def get_global_recorder():
    return global_recorder

def set_global_recorder(recorder:TokenRecords):
    global global_recorder
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
        token, state, toks = extract_info(parser)
        global_recorder.add_state_record(parser, token, before_state, state, toks)
        if is_debug:
            print('end monitor all: ', before_state, atnState)
        return res
    return record_state


class MonitorParser(CParser):
    @property
    def state(self):
        return self._stateNumber

    @state.setter
    @record_state_wrapper
    def state(self, atnState: int):
        self._stateNumber = atnState


# ------------------------------------------- create parser-sync listener --------------------------------- #
def create_register_enter_listener_parser(parser):
    parser = register_parser_listener(RecordEnterListener, enter_all_fn, parser)
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
    token, state, toks = extract_info(parser)
    global_recorder.add_rule_record(parser, token, state, toks)
    if is_debug:
        print('end enter: ', parser.state)


class RecordEnterListener(CListener):
    pass

class ExceptionErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        full_msg = "syntaxError: line " + str(line) + ":" + str(column) + " " + msg
        raise Exception(full_msg)
