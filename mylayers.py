import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from util import masked_softmax, masked_max


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
#        a_i = F.dropout(a_i, self.drop_prob, self.training)

        return a_i # logits


class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, drop_prob=0., end_of_seq=False, use_gru=True):
        super(RNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.use_gru = use_gru
        self.num_layers = num_layers
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

        if self.end_of_seq:
            # h_n (num_layers*2, batch, self.hidden_size)
            h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)
            # h_n (num_layers, 2, batch, self.hidden_size)
            # picking the upper layer
            h_n = h_n[-1, :, :, :].squeeze(0)
            # h_n (2, batch, self.hidden_size)

            out = h_n.permute(1, 0, 2).contiguous().view((-1, self.hidden_size*2)).unsqueeze(-2)
            out = out[unsort_idx]
            # out (batch_size, 1, 2*self.hidden_size)
        else:
            out, _ = pad_packed_sequence(output, batch_first=True, total_length=padded_len)
            out = F.dropout(out, self.drop_prob, self.training)
            out = out[unsort_idx]
            # out (batch_size, c_len, 2*self.hidden_size)

        return out

class AttentionFlowLayer(nn.Module):
    def __init__(self, input_size, drop_prob=0.):
        super(AttentionFlowLayer, self).__init__()
        self.hidden_size = input_size
        self.drop_prob = drop_prob

        self.sim_proj = nn.Linear(3 * input_size, 1, bias=True)

    def similarity_matrix(self, h, u):
        # h (batch, h_len, input_size)
        # u (batch, u_len, input_size)
        h_dot_u = torch.mul(h.unsqueeze(2), u.unsqueeze(1)) # (batch, h_len, u_len, input_size)
        h_len, u_len = h.size(1), u.size(1)
        h_mat = h.unsqueeze(2).repeat(1, 1, u_len, 1)       # (batch, h_len, u_len, input_size)
        u_mat = u.unsqueeze(1).repeat(1, h_len, 1, 1)       # (batch, h_len, u_len, input_size)
        sim_input = torch.cat((h_mat, u_mat, h_dot_u), dim=-1)  # (batch, h_len, u_len, 3 * input_size)
        S = self.sim_proj(sim_input)                        # (batch, h_len, u_len, 1)
        S = S.squeeze(-1)
        return S

    def attention_vectors(self, h, u, h_mask, u_mask):
        # h_mask (batch, h_len)
        # u_mask (batch, u_len)
        S = self.similarity_matrix(h, u)
        h_len, u_len = h.size(1), u.size(1)

        # Context-to-query Attention
        u_mask = u_mask.view(-1, 1, u_len)
        a_t = masked_softmax(S, u_mask, dim=2)               # (batch, h_len, u_len)
        U_tilde = torch.bmm(a_t, u)                 # (batch, h_len, input_size)

        # # Query-to-context Attention
        # b = masked_softmax(masked_max(S, u_mask, dim=2), h_mask, dim=1) # (batch, h_len)
        # b = b.unsqueeze(dim=1)                      # (batch, 1, h_len)
        # H_tilde = torch.bmm(b, h)                   # (batch, 1, input_size)
        # H_tilde = H_tilde.repeat(1, h_len, 1)       # (batch, h_len, input_size)

        # Query-to-context Attention (Stanford variant)
        h_mask = h_mask.view(-1, h_len, 1)
        b_t = masked_softmax(S, h_mask, dim=1)      # (batch, h_len, u_len)
        b_t = torch.bmm(b_t, a_t.transpose(1, 2))   # (batch, h_len, h_len)
        H_tilde = torch.bmm(b_t, h)                 # (batch, h_len, input_size)

        return H_tilde, U_tilde

    def forward(self, h, u, h_mask, u_mask):
        h = F.dropout(h, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        u = F.dropout(u, self.drop_prob, self.training)  # (bs, q_len, hid_size)
        H_tilde, U_tilde = self.attention_vectors(h, u, h_mask, u_mask)
        out = torch.cat((h, U_tilde, h * U_tilde, h * H_tilde), dim=-1)  # (batch, h_len, 4 * input_size)

        return out

class BiDAFOutputLayer(nn.Module):
    def __init__(self, hidden_size, drop_prob=0., use_gru=True):
        super(BiDAFOutputLayer, self).__init__()
        self.hidden_size = hidden_size
        self.p1_proj = nn.Linear(10*hidden_size, 1)
        self.rnn_p2 = RNNLayer(2*hidden_size, hidden_size,
                               num_layers=1, drop_prob=drop_prob,
                               end_of_seq=False, use_gru=use_gru)
        self.p2_proj = nn.Linear(10*hidden_size, 1)

    def forward(self, g, m, masks):
        lengths = masks.sum(-1)

        p1_input = torch.cat((g, m), dim=-1)        # (batch, h_len, 10*hidden_size)
        p1 = self.p1_proj(p1_input).squeeze(-1)     # (batch, h_len)

        m2 = self.rnn_p2(m, lengths)

        p2_input = torch.cat((g, m2), dim=-1)        # (batch, h_len, 10*hidden_size)
        p2 = self.p2_proj(p2_input).squeeze(-1)      # (batch, h_len)

        log_p1 = masked_softmax(p1, masks, log_softmax=True)
        log_p2 = masked_softmax(p2, masks, log_softmax=True)

        return log_p1, log_p2


class HighwayNetwork(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(HighwayNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(EmbeddingLayer, self).__init__()
        self.drop_prob = drop_prob
        _, self.input_size = word_vectors.size()
        self.hidden_size = hidden_size

        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(self.input_size, hidden_size, bias=False)
        self.highway = HighwayNetwork(2, hidden_size=hidden_size)

    def forward(self, idxs):
        emb = self.embed(idxs)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)
        emb = self.highway(emb)

        return emb


