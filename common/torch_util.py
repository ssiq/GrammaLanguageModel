import torch

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