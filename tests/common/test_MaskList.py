import unittest

import torch

from common.util import generate_mask


class MaskList(unittest.TestCase):
    def setUp(self):
        self._mask_list = generate_mask([1, (4, 8), 9], 10)
        self._list = [0, 1, 0, 0, 1, 1, 1, 1, 1, 1]

    def test_get_item(self):
        for i in range(len(self._mask_list)):
            self.assertEqual(self._list[i], self._mask_list[i])

    def test_iter(self):
        self.assertEqual(self._list, self._mask_list)

    def test_len(self):
        self.assertEqual(len(self._list), len(self._mask_list))

    def test_parse_tensor(self):
        print(torch.LongTensor(self._mask_list))