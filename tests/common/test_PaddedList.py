from common.util import PaddedList, ShapeDifferentException, ListShapeErrorException

import unittest
import torch

class TestPaddedList(unittest.TestCase):

    def setUp(self):
        self.a = [1, 2, 3]
        self.ap = PaddedList(self.a)
        self.b = [[[1, 11, 111], [2]], [[4, 44], [5, 55, 555, 5555], [6]], [[6, 66]]]
        self.bp = PaddedList(self.b)

    def test_l_shape(self):
        res = self.ap._l_shape([1, 2, 3])
        self.assertEquals(res, [3])
        res = self.ap._l_shape([[1, 2, 3], [4, 5], [6]])
        self.assertEquals(res, [3, 3])
        res = self.ap._l_shape(((1, 2, 3), (4, 5), (6, )))
        self.assertEquals(res, [3, 3])
        res = self.ap._l_shape([[1, 2], [4, 5, 6], [6]])
        self.assertEquals(res, [3, 3])
        res = self.ap._l_shape([[[1, 11, 111], [2]], [[4, 44], [5, 55, 555, 5555], [6]], [[6, 66]]])
        self.assertEquals(res, [3, 3, 4])

    def test_init(self):
        self.assertEquals(self.bp.l, self.b)
        self.assertEquals(self.bp.shape, [3, 3, 4])
        self.assertEquals(self.bp.fill_value, 0)

    def test_cal_max_shapes(self):
        shape1 = [2, 2, 3]
        shape2 = [1, 3, 4]
        res = self.ap._cal_max_shapes(shape1, shape2)
        self.assertEquals(res, [2, 3, 4])
        shape3 = [2, 2]
        self.assertRaises(ShapeDifferentException, self.ap._cal_max_shapes, shape1, shape3)

    def test_create_list_as_shape(self):
        b = [[[1, 11, 111], [2]], [[4, 44], [5, 55, 555, 5555], [6]], [[6, 66]]]
        target1 = [
            [[1, 11, 111, 0],
             [2, 0, 0, 0],
             [0, 0, 0, 0]],
            [[4, 44, 0, 0],
             [5, 55, 555, 5555],
             [6, 0, 0, 0]],
            [[6, 66, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        ]
        res = self.ap._create_list_as_shape(b, shape=[3, 3, 4])
        self.assertEquals(res, target1)
        target2 = [
            [[1, 11, 111, 0, 0],
             [2, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            [[4, 44, 0, 0, 0],
             [5, 55, 555, 5555, 0],
             [6, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            [[6, 66, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
        ]
        res = self.ap._create_list_as_shape(b, shape=[4, 4, 5])
        self.assertEquals(res, target2)
        self.assertRaises(ListShapeErrorException, self.ap._create_list_as_shape, b, [5, 5, 5, 5])
        self.assertRaises(ListShapeErrorException, self.ap._create_list_as_shape, b, [2, 5, 5])

    def test_getitem(self):
        # test int item
        res = self.ap[1]
        self.assertEquals(2, res)
        res = self.bp[1]
        self.assertTrue(isinstance(res, PaddedList))
        self.assertEquals(res.shape, self.bp.shape[1:])
        res = self.bp[2][2]
        self.assertEquals(res.shape, self.bp.shape[2:])
        self.assertEquals(res.l, [])
        self.assertEquals(res.to_list(), [0, 0, 0, 0])
        self.assertEquals(res[0], 0)
        # test slice item
        res = self.ap[1:]
        self.assertTrue(isinstance(res, PaddedList))
        self.assertEquals(res.shape, [2])
        self.assertEquals(res.to_list(), [2, 3])
        res = self.ap[:-1]
        self.assertTrue(isinstance(res, PaddedList))
        self.assertEquals(res.shape, [2])
        self.assertEquals(res.to_list(), [1, 2])
        res = self.ap[1: 2: 1]
        self.assertEquals(res.shape, [1])
        self.assertEquals(res.to_list(), [2])
        res = self.ap[2: 1: -1]
        self.assertEquals(res.shape, [1])
        self.assertEquals(res.to_list(), [3])
        res = self.bp[1:]
        target1 = [
            [[4, 44, 0, 0],
             [5, 55, 555, 5555],
             [6, 0, 0, 0]],
            [[6, 66, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        ]
        self.assertEquals(res.shape, [2, 3, 4])
        self.assertEquals(res.to_list(), target1)

    def test_len(self):
        self.assertEquals(len(self.ap), 3)
        self.assertEquals(len(self.bp), 3)
        self.assertEquals(len(self.bp[2][2]), 4)

    def test_iter(self):
        res = []
        target1 = [1, 2, 3]
        for i in self.ap:
            res += [i]
        for r, t in zip(res, target1):
            self.assertEquals(r, t)
        res = []
        target2 = [PaddedList([[1, 11, 111], [2]], shape=[3, 4]),
                   PaddedList([[4, 44], [5, 55, 555, 5555], [6]], shape=[3, 4]),
                   PaddedList([[6, 66]], shape=[3, 4])]
        for i in self.bp:
            res += [i]
        for r, t in zip(res, target2):
            self.assertEquals(r.shape, t.shape)
            self.assertEquals(r.l, t.l)
            self.assertEquals(r.to_list(), t.to_list())

    def test_reverse(self):
        res = []
        target1 = [1, 2, 3]
        for i in reversed(self.ap):
            res = [i] + res
        for r, t in zip(res, target1):
            self.assertEquals(r, t)
        res = []
        target2 = [PaddedList([[1, 11, 111], [2]], shape=[3, 4]),
                   PaddedList([[4, 44], [5, 55, 555, 5555], [6]], shape=[3, 4]),
                   PaddedList([[6, 66]], shape=[3, 4])]
        for i in reversed(self.bp):
            res = [i] + res
        for r, t in zip(res, target2):
            self.assertEquals(r.shape, t.shape)
            self.assertEquals(r.l, t.l)
            self.assertEquals(r.to_list(), t.to_list())

    def test_contains(self):
        self.assertTrue(2 in self.ap)
        self.assertTrue(1 in self.ap)
        cp = PaddedList([1, 2, 3], shape=[4])
        self.assertTrue(0 in cp)
        self.assertTrue(PaddedList([[6, 66]], shape=[3, 4]) in self.bp)

    def test_index(self):
        res = self.bp.index(PaddedList([[6, 66]], shape=[3, 4]))
        self.assertEquals(res, 2)
        cp = PaddedList([1, 2, 3], shape=[4])
        res = cp.index(0)
        self.assertEquals(res, 3)
        res = cp.index(5)
        self.assertEquals(res, -1)

    def test_count(self):
        res = self.bp.count(PaddedList([[6, 66]], shape=[3, 4]))
        self.assertEquals(res, 1)
        res = self.bp.count(PaddedList([[6, 666]], shape=[3, 4]))
        self.assertEquals(res, 0)
        cp = PaddedList([1, 2, 3], shape=[6])
        res = cp.count(0)
        self.assertEquals(res, 3)
        res = cp.count(5)
        self.assertEquals(res, 0)

    def test_equals(self):
        dp = PaddedList([[6, 66]], shape=[3, 4])
        self.assertTrue(dp == self.bp[2])
        self.assertFalse(dp != self.bp[2])
        dp = PaddedList([[6, 666]], shape=[3, 4])
        self.assertFalse(dp == self.bp[2])
        self.assertTrue(dp != self.bp[2])

    def test_tensor(self):
        ten = torch.Tensor(self.ap)
        print(ten)
        ten = torch.Tensor(self.bp)
        print(ten)
        ep = PaddedList([[4, 44], [5, 55, 555, 5555], [6]], shape=[3, 4])
        ten = torch.Tensor(ep)
        print(ten)


