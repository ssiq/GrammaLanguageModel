import os
import sys

import numpy
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import pandas as pd
from torch.utils.data import Dataset

import config
from common import torch_util
from common.torch_util import padded_tensor_one_dim_to_length
from common.util import data_loader, show_process_map, PaddedList, convert_one_token_ids_to_code, compile_c_code_by_gcc
from embedding.wordembedding import Vocabulary, load_vocabulary
from read_data.load_parsed_data import get_random_error_c99_code_token_vocabulary, \
    get_random_error_c99_code_token_vocabulary_id_map, get_common_error_c99_code_token_vocabulary, \
    get_common_error_c99_code_token_vocabulary_id_map, generate_tokens_for_c_error_dataset
from read_data.read_experiment_data import read_fake_random_c_error_dataset, read_fake_common_c_error_dataset

available_cuda = True
GPU_INDEX = 0
MAX_LENGTH = 500
BEGIN, END, UNK = ["<BEGIN>", "<END>", "<UNK>"]

is_debug=False

TARGET_PAD_TOKEN = -1


def trans_to_cuda(x):
    if available_cuda:
        return x.cuda(GPU_INDEX)
    return x


class CCodeErrorDataSet(Dataset):
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
        error_tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["tokens"]]],
                                                              use_position_label=True)[0]
        ac_tokens = self.vocabulary.parse_text_without_pad([[k.value for k in self.data_df.iloc[index]["ac_tokens"]]],
                                                              use_position_label=True)[0]

        sample = {"error_tokens": error_tokens,
                  'error_length': len(error_tokens),
                  "ac_tokens_input": ac_tokens,
                  "ac_length": len(ac_tokens),
                  'target_tokens': ac_tokens[1:],
                  'includes': self.data_df.iloc[index]['includes']}
        return sample

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_dim, vocab_size, rnn_layer_num, batch_size, dropout_p, is_bidirectional=True):
        super(EncoderRNN, self).__init__()
        self._hidden_size = hidden_size
        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._rnn_layer_num = rnn_layer_num
        self._batch_size = batch_size
        self._dropout_p = dropout_p
        self._is_bidirectional = is_bidirectional
        self.bidirectional_num = 2 if is_bidirectional else 1

        self.encoder_embedding = trans_to_cuda(nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0))
        self.lstm = trans_to_cuda(nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=rnn_layer_num,
                            bidirectional=is_bidirectional, dropout=dropout_p, batch_first=True))
        self.drop = trans_to_cuda(nn.Dropout(dropout_p))


    def init_hidden(self):
        hidden = (torch.autograd.Variable(torch.randn(self._rnn_layer_num * self.bidirectional_num, self._batch_size, self._hidden_size)),
                torch.autograd.Variable(torch.randn(self._rnn_layer_num * self.bidirectional_num, self._batch_size, self._hidden_size)))
        hidden = [trans_to_cuda(one) for one in hidden]
        return hidden

    def forward(self, inputs, token_length):

        _, idx_sort = torch.sort(token_length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        inputs = torch.index_select(inputs, 0, idx_sort)
        sorted_token_length = list(torch.index_select(token_length, 0, idx_sort))

        hidden = self.init_hidden()

        print('input_size: ', inputs.size())
        inputs = trans_to_cuda(autograd.Variable(inputs))
        embedded = trans_to_cuda(self.encoder_embedding(inputs).view(self._batch_size, -1, self._embedding_dim))
        drop_embedded = trans_to_cuda(self.drop(embedded))

        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(drop_embedded, sorted_token_length, batch_first=True)
        lstm_out, hidden = self.lstm(packed_inputs, hidden)
        unpacked_lstm_out, unpacked_lstm_length = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True,
                                                                                         padding_value=0)

        unpacked_out = torch.index_select(unpacked_lstm_out, 0, trans_to_cuda(autograd.Variable(idx_unsort)))

        hidden = [torch.index_select(hid, 1, trans_to_cuda(autograd.Variable(idx_unsort))) for hid in hidden]

        return unpacked_out, hidden


class Attention(nn.Module):

    def __init__(self, hidden_size, embedding_dim, sequence_max_length, batch_size):
        super(Attention, self).__init__()
        self._hidden_size = hidden_size
        self._embedding_dim = embedding_dim
        self._sequence_max_length = sequence_max_length
        self._batch_size = batch_size

        # self.attn = trans_to_cuda(nn.Linear(hidden_size + embedding_dim, sequence_max_length))
        # self.attn_combine = trans_to_cuda(nn.Linear(hidden_size + embedding_dim, hidden_size))
        self.linear_out = trans_to_cuda(nn.Linear(2 * hidden_size, hidden_size))

    # def forward(self, embedded, hidden, encoder_outputs, mask):
    #     # encoder_outputs size = [batch_size, max_len, hidden_size]
    #
    #     embedded = embedded.view(self._batch_size, 1, self._embedding_dim)
    #
    #     # torch.cat((embedded, hidden[0]), dim=2)
    #     trans_hidden = torch.transpose(hidden[0], 0, 1)
    #     attn_value = self.attn(torch.cat((embedded, trans_hidden), dim=2))
    #     attn_value = attn_value * mask.unsqueeze(1)
    #
    #     attn_weights = F.softmax(attn_value, dim=2)     # size= [batch_size, 1, max_len]
    #     attn_applied = torch.bmm(attn_weights, encoder_outputs)     # size = [batch_size, 1, hidden_size]
    #
    #     output = torch.cat((embedded, attn_applied), dim=2)
    #     output = self.attn_combine(output)
    #     output = F.relu(output)
    #     return output, attn_weights

    def ibm_forward(self, output, context, mask):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if mask is not None:
            attn.data.masked_fill_(mask.unsqueeze(1), -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        return output, attn


class DecoderRNNCell(nn.Module):
    def __init__(self, hidden_size, embedding_dim, vocab_size, rnn_layer_num, batch_size, dropout_p, sequence_max_length, end_token, is_bidirectional=True):
        super(DecoderRNNCell, self).__init__()
        self._hidden_size = hidden_size
        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._rnn_layer_num = rnn_layer_num
        self._batch_size = batch_size
        self._dropout_p = dropout_p
        self._is_bidirectional = is_bidirectional
        self._sequence_max_length = sequence_max_length
        self.bidirectional_num = 2 if is_bidirectional else 1
        self.end_token = end_token

        self.decoder_embedding = trans_to_cuda(nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0))
        self.attention = Attention(hidden_size=hidden_size, embedding_dim=embedding_dim, sequence_max_length=sequence_max_length, batch_size=batch_size)

        # self.lstm = trans_to_cuda(nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=rnn_layer_num,
        #                                   bidirectional=is_bidirectional, dropout=dropout_p, batch_first=True))
        self.lstm_ibm = trans_to_cuda(nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=rnn_layer_num,
                                          bidirectional=is_bidirectional, dropout=dropout_p, batch_first=True))
        self.dropout = trans_to_cuda(nn.Dropout(dropout_p))
        self.out = trans_to_cuda(nn.Linear(self._hidden_size, self._vocab_size))


    # def _decoder_cell_forward(self, one_input, hidden, encoder_outputs, mask):
    #     embedded = trans_to_cuda(self.decoder_embedding(one_input)).view(self._batch_size, 1, self._embedding_dim)
    #     embedded = trans_to_cuda(self.dropout(embedded))
    #
    #     attn_output, attn_weights = self.attention.forward(embedded, hidden, encoder_outputs, mask)
    #     lstm_output, hidden = self.lstm(attn_output, hidden)
    #
    #     output = self.out(lstm_output)
    #
    #     return output, hidden, attn_weights

    def _forward_step(self, decoder_input, hidden, encoder_outputs, mask):
        output_size = list(decoder_input.shape)[1]
        embedded = trans_to_cuda(self.decoder_embedding(decoder_input)).view(self._batch_size, output_size,
                                                                             self._embedding_dim)
        embedded = trans_to_cuda(self.dropout(embedded))
        hidden = [hid.contiguous() for hid in hidden]
        output, hidden = self.lstm_ibm(embedded, hidden)

        output, attn = self.attention.ibm_forward(output, encoder_outputs, mask)
        predicted_softmax = F.log_softmax(self.out(output.contiguous().view(-1, self._hidden_size)), dim=1).view(
            self._batch_size,
            output_size,
            -1)
        return predicted_softmax, hidden, attn

    def _decoder_forward(self, inputs, hidden, encoder_outputs, mask):
        decoder_input = inputs[:, :-1]
        predicted_softmax, hidden, attn = self._forward_step(decoder_input, hidden, encoder_outputs, mask)
        return predicted_softmax, hidden, attn

    def _test_decoder_forward(self, inputs, hidden, encoder_outputs, mask):
        decoder_input = inputs[:, 0].unsqueeze(1).long()

        outputs = trans_to_cuda(torch.zeros([self._batch_size, self._sequence_max_length, self._vocab_size]))
        # is_continue = torch.ones([self._batch_size]).byte()
        # is_continue = numpy.array([1] * self._batch_size)
        # output_length = numpy.array([self._sequence_max_length] * self._batch_size)

        is_continue = trans_to_cuda(torch.ones([self._batch_size]).byte())
        output_length = trans_to_cuda(torch.LongTensor([self._sequence_max_length] * self._batch_size))

        for i in range(self._sequence_max_length):
            predicted_softmax, hidden, attn = self._forward_step(decoder_input, hidden, encoder_outputs, mask)
            outputs[:, i, :] = predicted_softmax[:, 0, :]
            decoder_input = predicted_softmax.topk(1)[1]

            end_batches = decoder_input.data.eq(self.end_token).view(-1)
            # end_batches = end_batches.cpu().view(-1).numpy()
            # update_idx = ((self._sequence_max_length > i) & end_batches & is_continue) != 0
            if self._sequence_max_length > i:
                update_idx = (end_batches & is_continue)
                output_length[update_idx] = i+1
                is_continue[update_idx] = 0
        return outputs, hidden, attn


class Seq2SeqModel(nn.Module):

    def __init__(self, hidden_size, embedding_dim, vocab_size, rnn_layer_num, batch_size, dropout_p, sequence_max_length, start_token, end_token, encoder_is_bidirectional=True, decoder_is_bidirectional=False):
        super(Seq2SeqModel, self).__init__()
        self._hidden_size = hidden_size
        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._rnn_layer_num = rnn_layer_num
        self._batch_size = batch_size
        self._dropout_p = dropout_p
        self._sequence_max_length = sequence_max_length
        self._encoder_is_bidirectional = encoder_is_bidirectional
        self._encoder_bi_count = 2 if self._encoder_is_bidirectional else 1
        self._decoder_is_bidirectional = decoder_is_bidirectional
        self._decoder_bi_count = 2 if self._decoder_is_bidirectional else 1
        self._decoder_hidden_size = int(self._encoder_bi_count * hidden_size / self._decoder_bi_count)

        self._start_token = start_token
        self._end_token = end_token

        self.encode = EncoderRNN(hidden_size, embedding_dim, vocab_size, rnn_layer_num, batch_size, dropout_p, is_bidirectional=encoder_is_bidirectional)
        self.decode = DecoderRNNCell(self._decoder_hidden_size, embedding_dim, vocab_size, rnn_layer_num, batch_size, dropout_p, sequence_max_length, end_token, is_bidirectional=decoder_is_bidirectional)

    def _create_mask(self, token_length, max_len):
        idxes = torch.arange(0, max_len, out=torch.Tensor(max_len)).unsqueeze(0)  # some day, you'll be able to directly do this on cuda
        # mask = autograd.Variable((trans_to_cuda(idxes) < token_length.unsqueeze(1)).float())
        mask = (trans_to_cuda(idxes) < token_length.unsqueeze(1).float())
        return mask

    # def forward(self, encoder_inputs, encoder_input_length, decoder_inputs, decoder_input_length):
    #     # hidden size = [rnn_layer * bi_count, batch_size, hidden_size]
    #     encoder_outputs, hidden = self.encode.forward(encoder_inputs, encoder_input_length)  # encoder output size = [batch_size, max_len, bi_count*rnn_layer*hidden_size]
    #     encoder_outputs = padded_tensor_one_dim_to_length(encoder_outputs, dim=1, padded_length=self._sequence_max_length, is_cuda=available_cuda, gpu_index=GPU_INDEX)
    #     hidden = [torch.transpose(torch.transpose(hid, 0, 1).contiguous().view(self._batch_size, self._decoder_bi_count * self._rnn_layer_num, -1), 0, 1) for hid in hidden]
    #
    #     mask = self._create_mask(trans_to_cuda(encoder_input_length), self._sequence_max_length)
    #
    #     max_len = torch.max(decoder_input_length)
    #     output = torch.zeros([self._batch_size, max_len-1, self._vocab_size])
    #
    #     for i in range(max_len-1):
    #         one_input = decoder_inputs[:, i]
    #         one_output, hidden, attn_weights = self.decode_cell._decoder_cell_forward(one_input, hidden, encoder_outputs, mask)
    #         one_input = F.log_softmax(one_output, dim=2)
    #         output[:, i, :] = one_input[:, 0, :]
    #     return output

    def ibm_forward(self, encoder_inputs, encoder_input_length, decoder_inputs, decoder_input_length):
        # hidden size = [rnn_layer * bi_count, batch_size, hidden_size]
        encoder_outputs, hidden = self.encode.forward(encoder_inputs, encoder_input_length)  # encoder output size = [batch_size, max_len, bi_count*rnn_layer*hidden_size]
        encoder_outputs = padded_tensor_one_dim_to_length(encoder_outputs, dim=1, padded_length=self._sequence_max_length, is_cuda=available_cuda, gpu_index=GPU_INDEX)
        hidden = [torch.transpose(torch.transpose(hid, 0, 1).contiguous().view(self._batch_size, self._decoder_bi_count * self._rnn_layer_num, -1), 0, 1) for hid in hidden]

        mask = self._create_mask(trans_to_cuda(encoder_input_length), self._sequence_max_length)

        output, hidden, attn_weights = self.decode._decoder_forward(decoder_inputs, hidden, encoder_outputs, mask)
        return output

    def _ibm_test_forward(self, encoder_inputs, encoder_input_length):
        encoder_outputs, hidden = self.encode.forward(encoder_inputs, encoder_input_length)  # encoder output size = [batch_size, max_len, bi_count*rnn_layer*hidden_size]
        encoder_outputs = padded_tensor_one_dim_to_length(encoder_outputs, dim=1, padded_length=self._sequence_max_length, is_cuda=available_cuda, gpu_index=GPU_INDEX)
        hidden = [torch.transpose(torch.transpose(hid, 0, 1).contiguous().view(self._batch_size, self._decoder_bi_count * self._rnn_layer_num, -1), 0, 1) for hid in hidden]

        mask = self._create_mask(trans_to_cuda(encoder_input_length), self._sequence_max_length)

        decoder_inputs = trans_to_cuda(torch.Tensor([self._start_token for i in range(self._batch_size)]).unsqueeze(1))
        output, hidden, attn_weights = self.decode._test_decoder_forward(decoder_inputs, hidden, encoder_outputs, mask)
        return output

    # def test_forward(self, inputs, input_length):
    #     encoder_outputs, hidden = self.encode.forward(inputs, input_length)  # encoder output size = [batch_size, max_len, bi_count*rnn_layer*hidden_size]
    #     encoder_outputs = padded_tensor_one_dim_to_length(encoder_outputs, dim=1, padded_length=self._sequence_max_length, is_cuda=available_cuda, gpu_index=GPU_INDEX)
    #     hidden = [torch.transpose(
    #         torch.transpose(hid, 0, 1).contiguous().view(self._batch_size, self._decoder_bi_count * self._rnn_layer_num,
    #                                                      -1), 0, 1) for hid in hidden]
    #
    #     mask = self._create_mask(input_length, self._sequence_max_length)
    #
    #     output = torch.zeros([self._batch_size, self._sequence_max_length-1, self._vocab_size])
    #     is_continue = torch.ones([self._batch_size]).byte()
    #     output_length = torch.zeros([self._batch_size]).long()
    #
    #     one_input = trans_to_cuda(torch.Tensor([self._start_token for i in range(self._batch_size)]).long())
    #     for i in range(self._sequence_max_length-1):
    #         one_output, hidden, attn_weights = self.decode_cell._decoder_cell_forward(one_input, hidden, encoder_outputs, mask)
    #         one_output = one_output * is_continue.unsqueeze(1).unsqueeze(2).float()
    #         one_input_prob = F.log_softmax(one_output, dim=2)
    #         _, one_input = torch.max(one_input_prob, dim=2)
    #         one_continue = torch.ne(one_input.squeeze(1), self._end_token)
    #         one_end = torch.eq(one_input.squeeze(1), self._end_token)
    #         one_ture_end = one_end & is_continue
    #         output_length = torch.where(one_ture_end, torch.LongTensor([i+2 for t in range(self._batch_size)]), output_length)
    #
    #         is_continue = is_continue & one_continue
    #         output[:, i, :] = one_input_prob[:, 0, :]
    #     output_length = torch.where(is_continue, torch.LongTensor([self._sequence_max_length+1 for t in range(self._batch_size)]), output_length)
    #     return output, output_length


def train(model, dataset, batch_size, loss_function, optimizer):
    print('in train')
    total_loss = torch.Tensor([0])
    count = torch.Tensor([0])
    steps = 0
    model.train()
    for batch_data in data_loader(dataset, batch_size=batch_size, is_shuffle=True, drop_last=True):
        # with torch.autograd.profiler.profile() as prof:
        error_tokens = trans_to_cuda(torch.LongTensor(PaddedList(batch_data['error_tokens'])))
        error_length = trans_to_cuda(torch.LongTensor(batch_data['error_length']))
        ac_tokens_input = trans_to_cuda(torch.LongTensor(PaddedList(batch_data['ac_tokens_input'])))
        ac_tokens_length = trans_to_cuda(torch.LongTensor(batch_data['ac_length']))
        target_tokens = trans_to_cuda(torch.LongTensor(PaddedList(batch_data['target_tokens'], fill_value=TARGET_PAD_TOKEN)))


        del batch_data["error_tokens"], batch_data["error_length"], batch_data["ac_tokens_input"], batch_data["ac_length"], batch_data["target_tokens"]

        model.zero_grad()
        log_probs = model.ibm_forward(error_tokens, error_length, ac_tokens_input, ac_tokens_length)
        print('finish one step train')
        loss = loss_function(log_probs.view(log_probs.shape[0]*log_probs.shape[1], -1), target_tokens.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        print('finish optimizer step train')

        cur_target_count = (torch.sum(ac_tokens_length.data.cpu()) - batch_size).float()
        total_loss += (loss.data.cpu() * cur_target_count)
        count += cur_target_count
        steps += 1
        print('step {} loss: {}'.format(steps, loss))
        # print(prof)
        sys.stdout.flush()
        sys.stderr.flush()

    return (total_loss/count).data[0]

print_count = 0
def evaluate(model, dataset, loss_function, batch_size, start, end, unk, id_to_word_fn, file_path='test.c', use_force_train=False):
    print('in evaluate')
    global print_count
    steps = 0
    success = 0
    total = 0
    total_loss = torch.Tensor([0])
    count = torch.Tensor([0])
    total_correct = torch.Tensor([0])
    total_compare_correct = torch.Tensor([0])
    total_loss_in_train = torch.Tensor([0])
    count_in_train = torch.Tensor([0])
    model.eval()
    for batch_data in data_loader(dataset, batch_size=batch_size, is_shuffle=True, drop_last=True):
        error_tokens = trans_to_cuda(torch.LongTensor(PaddedList(batch_data['error_tokens'])))
        error_length = trans_to_cuda(torch.LongTensor(PaddedList(batch_data['error_length'])))
        ac_tokens_input = trans_to_cuda(torch.LongTensor(PaddedList(batch_data['ac_tokens_input'])))
        ac_tokens_length = trans_to_cuda(torch.LongTensor(PaddedList(batch_data['ac_length'])))
        target_tokens = trans_to_cuda(torch.LongTensor(PaddedList(batch_data['target_tokens'], fill_value=TARGET_PAD_TOKEN)))
        target_tokens_padded = padded_tensor_one_dim_to_length(target_tokens.float(), dim=1,
                                                          padded_length=MAX_LENGTH,
                                                          is_cuda=available_cuda, gpu_index=GPU_INDEX, fill_value=TARGET_PAD_TOKEN).long()
        del batch_data["error_tokens"], batch_data["error_length"], batch_data["ac_tokens_input"], batch_data[
            "ac_length"], batch_data["target_tokens"]
        includes = batch_data['includes']

        loss_in_train = None
        # calculate loss like train
        if use_force_train:
            log_probs = model.ibm_forward(error_tokens, error_length, ac_tokens_input, ac_tokens_length)
            print('finish one step train')
            loss_in_train = loss_function(log_probs.view(log_probs.shape[0] * log_probs.shape[1], -1), target_tokens.view(-1))

            cur_target_count = (torch.sum(ac_tokens_length.data.cpu()) - batch_size).float()
            total_loss_in_train += (loss_in_train.data.cpu() * cur_target_count)
            count_in_train += cur_target_count
        else:
            log_probs = model._ibm_test_forward(error_tokens, error_length)

        # do evaluate
        cur_batch_len = len(batch_data['includes'])

        predict_log_probs = torch.transpose(log_probs, 0, 1)
        target_label = torch.transpose(target_tokens_padded, 0, 1)
        cur_loss = torch.Tensor([0])
        cur_step = torch.Tensor([0])
        cur_correct = torch.Tensor([0])
        is_compare_success = torch.Tensor([1] * batch_size)
        for i, step_output in enumerate(predict_log_probs):
            step_target = target_label[i, :].view(batch_size)
            batch_loss = loss_function(step_output.view(batch_size, -1), step_target)
            batch_predict_label = step_output.view(batch_size, -1).topk(1)[1].view(batch_size)

            in_step_count = step_target.ne(TARGET_PAD_TOKEN).sum().float()
            cur_loss += (batch_loss.data.cpu() * in_step_count.cpu())
            cur_step += in_step_count.data.cpu()
            batch_correct = (step_target.ne(TARGET_PAD_TOKEN) & step_target.eq(batch_predict_label)).sum().cpu().float()
            batch_error = step_target.ne(TARGET_PAD_TOKEN) & step_target.ne(batch_predict_label)
            is_compare_success[batch_error.cpu()] = 0
            if batch_correct > 16:
                print(batch_correct)
            # batch_correct = (step_target.ne(TARGET_PAD_TOKEN) & step_target.eq(step_target)).sum().cpu().float()
            cur_correct += batch_correct
        total_loss += cur_loss
        total_correct += cur_correct
        count += cur_step
        total_compare_correct += is_compare_success.sum().float()

        _, output_tokens = torch.max(log_probs, dim=2)

        cur_success = 0
        for token_ids, include, ac_token_ids in zip(output_tokens, includes, ac_tokens_input):
            if print_count % 100 == 0:
                code = convert_one_token_ids_to_code(token_ids.tolist(), id_to_word_fn, start, end, unk, include)
                ac_code = convert_one_token_ids_to_code(ac_token_ids.tolist(), id_to_word_fn, start, end, unk, include)
                print(code)
                print(ac_code)
            # res = compile_c_code_by_gcc(code, file_path)
            res = False
            if res:
                cur_success += 1
        success += cur_success

        steps += 1
        total += cur_batch_len
        print_count += 1
        print('step {} accuracy: {}, loss: {}, correct: {}, compare correct: {}, loss according train: {}'.format(steps, cur_success/cur_batch_len, (cur_loss/cur_step).data[0], (cur_correct/cur_step).data[0], (is_compare_success.sum()/cur_batch_len).data[0], loss_in_train))
        sys.stdout.flush()
        sys.stderr.flush()
    return (total_loss/count).data[0], float(success/total), (total_correct/count).data[0], (total_compare_correct/total).data[0], (total_loss_in_train/count_in_train).data[0]


def train_and_evaluate(data, dataset_type, batch_size, embedding_dim, hidden_state_size, rnn_num_layer, dropout_p, learning_rate, epoches, saved_name, load_name=None, gcc_file_path='test.c', encoder_is_bidirectional=True, decoder_is_bidirectional=False):
    save_path = os.path.join(config.save_model_root, saved_name)
    if load_name is not None:
        load_path = os.path.join(config.save_model_root, load_name)

    for d, n in zip(data, ["train", "val", "test"]):
        print("There are {} raw data in the {} dataset".format(len(d), n))
    if dataset_type == 'random':
        vocabulary = load_vocabulary(get_random_error_c99_code_token_vocabulary, get_random_error_c99_code_token_vocabulary_id_map, [BEGIN], [END], UNK)
    elif dataset_type == 'common':
        vocabulary = load_vocabulary(get_common_error_c99_code_token_vocabulary, get_common_error_c99_code_token_vocabulary_id_map, [BEGIN], [END], UNK)
    else:
        vocabulary = load_vocabulary(get_common_error_c99_code_token_vocabulary, get_common_error_c99_code_token_vocabulary_id_map, [BEGIN], [END], UNK)
    generate_dataset = lambda df: CCodeErrorDataSet(df, vocabulary)
    data = [generate_dataset(d) for d in data]
    for d, n in zip(data, ["train", "val", "test"]):
        print("There are {} parsed data in the {} dataset".format(len(d), n))
    train_dataset, valid_dataset, test_dataset = data

    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    unk_id = vocabulary.word_to_id(vocabulary.unk)

    # loss_function = nn.CrossEntropyLoss(size_average=False, ignore_index=TARGET_PAD_TOKEN)
    loss_function = nn.CrossEntropyLoss(ignore_index=TARGET_PAD_TOKEN)
    model = Seq2SeqModel(hidden_state_size, embedding_dim, vocabulary.vocabulary_size, rnn_num_layer, batch_size, dropout_p, MAX_LENGTH, begin_id, end_id, encoder_is_bidirectional=encoder_is_bidirectional, decoder_is_bidirectional=decoder_is_bidirectional)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    if load_name is not None:
        torch_util.load_model(model, load_path)
        valid_loss, valid_accuracy, valid_correct, valid_compare_correct, _ = evaluate(model, valid_dataset, loss_function, batch_size, begin_id, end_id, unk_id, vocabulary.id_to_word, file_path=gcc_file_path)
        test_loss, test_accuracy, test_correct, test_compare_correct, _ = evaluate(model, test_dataset, loss_function, batch_size, begin_id, end_id, unk_id, vocabulary.id_to_word, file_path=gcc_file_path)
        _, _, _, _, valid_loss_in_train = evaluate(model, valid_dataset, loss_function, batch_size, begin_id, end_id, unk_id, vocabulary.id_to_word, file_path=gcc_file_path, use_force_train=True)
        _, _, _, _, test_loss_in_train = evaluate(model, test_dataset, loss_function, batch_size, begin_id, end_id, unk_id, vocabulary.id_to_word, file_path=gcc_file_path, use_force_train=True)
        best_valid_accuracy = valid_accuracy
        best_test_accuracy = test_accuracy
        best_valid_loss = valid_loss
        best_test_loss = test_loss
        best_valid_correct = valid_correct
        best_test_correct = test_correct
        best_valid_compare_correct = valid_compare_correct
        best_test_compare_correct = test_compare_correct
        best_valid_loss_in_train = valid_loss_in_train
        best_test_loss_in_train = test_loss_in_train
        # best_valid_accuracy = None
        # best_test_accuracy = None
        # best_valid_loss = None
        # best_test_loss = None
        # best_valid_correct = None
        # best_test_correct = None
        # best_valid_compare_correct = None
        # best_test_compare_correct = None
        # best_valid_loss_in_train = None
        # best_test_loss_in_train = None
        print(
            "load the previous mode, validation perplexity is {}, test perplexity is :{}".format(valid_accuracy, test_accuracy))
        scheduler.step(best_valid_loss)
    else:
        best_valid_accuracy = None
        best_test_accuracy = None
        best_valid_loss = None
        best_test_loss = None
        best_valid_correct = None
        best_test_correct = None
        best_valid_compare_correct = None
        best_test_compare_correct = None
        best_valid_loss_in_train = None
        best_test_loss_in_train = None

    for epoch in range(epoches):
        train_loss = train(model, train_dataset, batch_size, loss_function, optimizer)
        valid_loss, valid_accuracy, valid_correct, valid_compare_correct, _ = evaluate(model, valid_dataset, loss_function, batch_size, vocabulary.begin_tokens[0],
                                  vocabulary.end_tokens[0], vocabulary.unk, vocabulary.id_to_word,
                                  file_path=gcc_file_path)
        test_loss, test_accuracy, test_correct, test_compare_correct, _ = evaluate(model, test_dataset, loss_function, batch_size, vocabulary.begin_tokens[0], vocabulary.end_tokens[0],
                                 vocabulary.unk, vocabulary.id_to_word, file_path=gcc_file_path)
        _, _, _, _, valid_loss_in_train = evaluate(model, valid_dataset, loss_function, batch_size, begin_id, end_id,
                                                   unk_id, vocabulary.id_to_word, file_path=gcc_file_path,
                                                   use_force_train=True)
        _, _, _, _, test_loss_in_train = evaluate(model, test_dataset, loss_function, batch_size, begin_id, end_id,
                                                  unk_id, vocabulary.id_to_word, file_path=gcc_file_path,
                                                  use_force_train=True)

        # train_perplexity = torch.exp(train_loss)[0]

        scheduler.step(valid_loss)

        if best_valid_correct is None or valid_correct < best_valid_correct:
            best_valid_accuracy = valid_accuracy
            best_test_accuracy = test_accuracy
            best_valid_loss = valid_loss
            best_test_loss = test_loss
            best_valid_correct = valid_correct
            best_test_correct = test_correct
            best_valid_compare_correct = valid_compare_correct
            best_test_compare_correct = test_compare_correct
            best_valid_loss_in_train = valid_loss_in_train
            best_test_loss_in_train = test_loss_in_train
            if not is_debug:
                torch_util.save_model(model, save_path)

        print("epoch {}: train loss of {}, valid loss of {}, test loss of {},  "
              "valid accuracy of {}, test accuracy of {}, valid correct of {}, test correct of {}, "
              "valid compare correct of {}, test compare correct of {}, valid loss in train of {}, test loss in train of {}".
              format(epoch, train_loss, valid_loss, test_loss, valid_accuracy, test_accuracy, valid_correct, test_correct, valid_compare_correct, test_compare_correct, valid_loss_in_train, test_loss_in_train))
    print("The model {} best valid accuracy is {} and test accuracy is {} and "
          "best valid loss is {} and test loss is {}, best valid correct is {}, best test correct is {}"
          "best valid compare correct is {}, best test compare correct is {}, "
          "best valid loss in train is {} and best test loss in train is {}".
          format(saved_name, best_valid_accuracy, best_test_accuracy, best_valid_loss, best_test_loss, best_valid_correct, best_test_correct, best_valid_compare_correct, best_test_compare_correct, best_valid_loss_in_train, best_test_loss_in_train))


def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    data_type = 'common'
    if data_type == 'random':
        data = read_fake_random_c_error_dataset()
    else:
        data = read_fake_common_c_error_dataset()
    if is_debug:
        data = [df[:100] for df in data]

    data = generate_tokens_for_c_error_dataset(data)
    # train_and_evaluate(data, data_type, batch_size=2, embedding_dim=100, hidden_state_size=100, rnn_num_layer=1,
    #                    dropout_p=0, learning_rate=0.05, epoches=10, saved_name='neural_seq2seq_test.pkl',
    #                    load_name=None, gcc_file_path=os.path.join(config.temp_code_write_path, 'test.c'))
    train_and_evaluate(data, data_type, batch_size=16, embedding_dim=300, hidden_state_size=300, rnn_num_layer=2, dropout_p=0.2, learning_rate=0.005, epoches=10, saved_name='neural_seq2seq_1.pkl', load_name='neural_seq2seq_1.pkl', gcc_file_path=os.path.join(config.temp_code_write_path, 'test.c'))
    train_and_evaluate(data, data_type, batch_size=16, embedding_dim=200, hidden_state_size=200, rnn_num_layer=2, dropout_p=0.2, learning_rate=0.005, epoches=10, saved_name='neural_seq2seq_2.pkl', load_name=None, gcc_file_path=os.path.join(config.temp_code_write_path, 'test.c'))
    train_and_evaluate(data, data_type, batch_size=16, embedding_dim=100, hidden_state_size=100, rnn_num_layer=2, dropout_p=0.2, learning_rate=0.005, epoches=10, saved_name='neural_seq2seq_3.pkl', load_name=None, gcc_file_path=os.path.join(config.temp_code_write_path, 'test.c'))


if __name__ == '__main__':
    main()

    # model = Seq2SeqModel(10, 5, 20, 1, 4, 0, 10, 0, 19)
    #
    # encoder_inputs = [[0, 1, 2, 3, 4, 5, 19, 0],
    #                   [0, 2, 4, 6, 8, 10, 12, 19],
    #                   [0, 2, 4, 6, 8, 10, 12, 19],
    #                   [0, 2, 4, 6, 8, 10, 12, 19]]
    # encoder_input_length = [7, 8, 8, 8]
    # target_inputs = [[0, 15, 1, 2, 3, 4, 5, 19, 0],
    #                   [0, 15, 2, 4, 6, 8, 10, 12, 19],
    #                   [0, 15, 2, 4, 6, 8, 10, 12, 19],
    #                   [0, 15, 2, 4, 6, 8, 10, 12, 19]]
    # target_input_length = [8, 9, 9, 9]
    #
    # model.forward(torch.LongTensor(encoder_inputs), torch.LongTensor(encoder_input_length), torch.LongTensor(target_inputs), torch.LongTensor(target_input_length))
    # model.test_forward(torch.LongTensor(encoder_inputs), torch.LongTensor(encoder_input_length))