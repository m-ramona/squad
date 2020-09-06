from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):

    def __init__(self, hidden_size, drop_prob=0.):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.att_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, p_i, q_t):
        # q_t (batch_size, h, 1)
        p_i = self.att_proj(p_i)                # (batch_size, c_len, h)
        a_i = torch.bmm(p_i, q_t).squeeze(-1)   # (batch_size, c_len, 1)
        a_i = F.dropout(a_i, self.drop_prob, self.training)

        return a_i # logits


class AttentiveReaderOutput(nn.Module):

    def __init__(self, hidden_size):
        super(AttentiveReaderOutput, self).__init__()

    def forward(self, logits, mask, log_softmax=True):
        # logits (batch, seq_len)
        mask = mask.type(torch.float32)
        masked_logits = mask * logits + (1-mask) * -1e30
        loss_fn = F.log_softmax if log_softmax else F.softmax
        probs = loss_fn(masked_logits, dim=-1)

        return probs


class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, drop_prob=0., end_of_seq=False, use_gru=True):
        super(RNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.use_gru = use_gru
        rnn_init = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_init(input_size, hidden_size, num_layers,
                          dropout=drop_prob if num_layers > 1 else 0,
                          bidirectional=True)
        self.end_of_seq = end_of_seq

    def forward(self, x, lengths):
        # x (batch, sequence, emb)
        padded_len = x.size(1)

        lengths, sort_idx = lengths.sort(0, descending=True)
        _, unsort_idx = sort_idx.sort(0)
        x = x[sort_idx]
        x = pack_padded_sequence(x, lengths, batch_first=True)

        if self.use_gru:
            output, h_n = self.rnn(x)
        else:
            output, (h_n, c_n) = self.rnn(x)
        # h_n (num_layers*2, batch, self.hidden_size)

        if self.end_of_seq:
            out = h_n.permute(1, 0, 2).contiguous().view((-1, self.hidden_size*2)).unsqueeze(-2)
            out = F.dropout(out, self.drop_prob, self.training)
            out = out[unsort_idx]
            # out (batch_size, 1, 2*self.hidden_size)
        else:
            out, _ = pad_packed_sequence(output, batch_first=True, total_length=padded_len)
            out = F.dropout(out, self.drop_prob, self.training)
            out = out[unsort_idx]
            # out (batch_size, c_len, 2*self.hidden_size)

        return out


class AttentiveReaderModel(nn.Module):

    def __init__(self, word_vectors, hidden_size, drop_prob=0., use_gru=True, rnn_layers=1):
        super(AttentiveReaderModel, self).__init__()
        self.vocab, self.input_size = word_vectors.size()
        self.hidden_size = hidden_size
        self.h = 2 * hidden_size
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.passage_rnn = RNNLayer(self.input_size, self.hidden_size, rnn_layers,
                                    drop_prob=drop_prob,
                                    end_of_seq=False,
                                    use_gru=use_gru)
        self.query_rnn = RNNLayer(self.input_size, self.hidden_size, rnn_layers,
                                  drop_prob=drop_prob,
                                  end_of_seq=True,
                                  use_gru=use_gru)
        self.att_start = AttentionLayer(self.h, drop_prob=drop_prob)
        self.att_end = AttentionLayer(self.h, drop_prob=drop_prob)
        self.output = AttentiveReaderOutput(self.h)

    def forward(self, cw_idxs, qw_idxs):
        # cw_idxs (batch_size, c_len)
        # qw_idxs (batch_size, q_len)

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.embed(cw_idxs)         # (batch_size, c_len, input_size)
        q_emb = self.embed(qw_idxs)         # (batch_size, q_len, input_size)

        p_i = self.passage_rnn(c_emb, c_len)    # (batch_size, c_len, h)
        q = self.query_rnn(q_emb, q_len)        # (batch_size, 1, h)
        q_t = q.permute(0, 2, 1)                # (batch_size, h, 1)

        logits_start = self.att_start(p_i, q_t)
        logits_end = self.att_end(p_i, q_t)

        probs_start = self.output(logits_start, c_mask, log_softmax=True)   # (batch_size, c_len)
        probs_end = self.output(logits_end, c_mask, log_softmax=True)       # (batch_size, c_len)

        return probs_start, probs_end