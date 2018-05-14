import operator
from collections import OrderedDict
from itertools import islice

import torch
import typing
from torch.nn.modules.rnn import RNNCellBase


def save_model(model: torch.nn.Module, path):
    torch.save(model.state_dict(), path)

def load_model(model: torch.nn.Module, path):
    model.load_state_dict(torch.load(path))

def mask_softmax(logit, mask):
    logit = logit * mask
    logit_max, _ = torch.max(logit, dim=-1, keepdim=True)
    logit = logit - logit_max
    logit_exp = torch.exp(logit) * mask
    softmax = logit_exp/torch.sum(logit_exp, dim=-1, keepdim=True)
    return softmax


def to_sparse(x, cuda=True, gpu_index=0):
    """ converts dense tensor x to sparse format """
    print(torch.typename(x))
    x_typename = torch.typename(x).split('.')[-1]
    if cuda:
        sparse_tensortype = getattr(torch.cuda.sparse, x_typename)
    else:
        sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    if cuda:
        return sparse_tensortype(indices, values, x.size(), device=torch.device('cuda:{}'.format(gpu_index)))
    else:
        return sparse_tensortype(indices, values, x.size())


def pack_padded_sequence(padded_sequence, length, batch_firse=False,GPU_INDEX=0):
    _, idx_sort = torch.sort(length, dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)
    length = torch.index_select(length, 0, idx_sort)
    if padded_sequence.is_cuda:
        padded_sequence = torch.index_select(padded_sequence, 0, idx_sort.cuda(GPU_INDEX))
    else:
        padded_sequence = torch.index_select(padded_sequence, 0, idx_sort)
    return torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, list(length), batch_first=batch_firse), idx_unsort


def pad_packed_sequence(packed_sequence, idx_unsort, pad_value, batch_firse=False, GPU_INDEX=0):
    padded_sequence, length = torch.nn.utils.rnn.pad_packed_sequence(packed_sequence, batch_first=batch_firse,
                                                                padding_value=pad_value)
    if padded_sequence.is_cuda:
        return torch.index_select(padded_sequence, 0, torch.autograd.Variable(idx_unsort).cuda(GPU_INDEX)), length
    else:
        return torch.index_select(padded_sequence, 0, torch.autograd.Variable(idx_unsort)), length


def pack_sequence(sequences, GPU_INDEX=0):
    length = torch.Tensor([len(seq) for seq in sequences])
    _, idx_sort = torch.sort(length, dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)
    sequences = sorted(sequences, key=lambda x: len(x), reverse=True)
    packed_sequences = torch.nn.utils.rnn.pack_sequence(sequences)
    return packed_sequences, idx_unsort


def create_ori_index_to_packed_index_dict(batch_sizes):
    begin_index = 0
    end_index = 0
    res = {}
    for i in range(len(batch_sizes)):
        end_index += batch_sizes[i]
        for j in range(end_index-begin_index):
            res[(j, i)] = begin_index + j
        begin_index += batch_sizes[i]
    return res


def create_stable_log_fn(epsilon):
    def stable_log(softmax_value):
        softmax_value = torch.clamp(softmax_value, epsilon, 1.0-epsilon)
        return torch.log(softmax_value)
    return stable_log


def padded_tensor_one_dim_to_length(one_tensor, dim, padded_length, is_cuda=False, gpu_index=0, fill_value=0):
    before_encoder_shape = list(one_tensor.shape)
    before_encoder_shape[dim] = padded_length - before_encoder_shape[dim]
    expend_tensor = (torch.ones(before_encoder_shape) * fill_value)
    if is_cuda:
        expend_tensor = expend_tensor.cuda(gpu_index)
    padded_outputs = torch.cat((one_tensor, expend_tensor), dim=dim)
    return padded_outputs


class MultiRNNCell(RNNCellBase):
    def __init__(self, cell_list: typing.List[RNNCellBase]):
        super().__init__()
        for idx, module in enumerate(cell_list):
            self.add_module(str(idx), module)

    def reset_parameters(self):
        for cell in self._modules.values():
            cell.reset_parameters()

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return MultiRNNCell(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, h_i, h_s):
        res_h = []
        for h, cell in zip(h_s, self._modules.values()):
            h = cell(h_i, h)
            res_h.append(h)
            if isinstance(cell, torch.nn.LSTMCell):
                h_i = h[0]
            else:
                h_i = h
        return h_i, res_h

