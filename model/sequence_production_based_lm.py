import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms, utils

import pandas as pd

from c_code_processer.code_util import LeafToken, MonitoredParser, parse_tree_to_top_down_process
from common.util import show_process_map, key_transform, FlatMap
from embedding.wordembedding import Vocabulary
from read_data.load_parsed_data import read_parsed_top_down_code

BEGIN, END, UNK = ["<BEGIN>", "<END>", "<UNK>"]
BEGIN_PRODUCTION, END_PRODUCTION = ['<BEGIN_PRODUCTION>', '<END_PRODUCTION>',]
MAX_LENGTH = 500


class CCodeDataSet(Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary: Vocabulary,
                 transform=None):
        self.data_df = data_df[data_df['tokens'].map(lambda x: x is not None)]
        self.data_df = self.data_df[self.data_df['tokens'].map(lambda x: len(x) < MAX_LENGTH)]
        self.transform = transform
        self.vocabulary = vocabulary

        self._samples = [self._get_raw_sample(i) for i in range(len(self.data_df))]
        if self.transform:
            self._samples = show_process_map(self.transform, self._samples)
        # for s in self._samples:
        #     for k, v in s.items():
        #         print("{}:shape {}".format(k, np.array(v).shape))

    def _get_raw_sample(self, index):
        tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["tokens"]]],
                                                        use_position_label=True)[0]
        sample = {"tree": self.data_df.iloc[index]["parse_tree"],
                  "tokens": tokens[:-1],
                  "target": tokens[1:],
                  "length": len(tokens)-1}
        return sample

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)


class ProductionSequenceMap(object):
    def __call__(self, sample):
        """
        :param sample: a Parse Node List
        :return: a list of list. The inner list contains a sub process from
        the time of one Terminal token on the top of stack to another time like this
        """
        res = []
        cache = []
        for node in sample:
            if isinstance(node, LeafToken):
                # print("Terminal token:{}".format(node.value))
                res.append(cache)
                cache = []
            else:
                cache.append(node)
        if not cache:
            print("There has something left in the cache")
            res.append(cache)
        return res


class ProductionIdMap(object):
    def __call__(self, sample):
        pass


if __name__ == '__main__':
    # data = read_parsed_top_down_code(True)
    monitor = MonitoredParser(lex_optimize=False,
                              yacc_debug=True,
                              yacc_optimize=False,
                              yacctab='yacctab')
    code = """
          int add(int a, int b)
          {
              return a+b*c;
          }
          """
    node, _, tokens = monitor.parse_get_production_list_and_token_list(code)
    productions = parse_tree_to_top_down_process(node)
    transforms_fn = transforms.Compose([
        key_transform(ProductionSequenceMap(), "tree"),
        FlatMap(),
    ])