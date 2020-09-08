import torch
import torch.nn as nn
import torch.nn.functional as F

from mylayers import AttentionLayer, RNNLayer
from util import masked_softmax


class AttentiveReaderModel(nn.Module):

    def __init__(self, word_vectors, hidden_size, drop_prob=0., rnn_layers=1, use_gru=True):
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
        self.mod_rnn = None

    def forward(self, cw_idxs, qw_idxs):
        # cw_idxs (batch_size, c_len)
        # qw_idxs (batch_size, q_len)

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.embed(cw_idxs)         # (batch_size, c_len, input_size)
        q_emb = self.embed(qw_idxs)         # (batch_size, q_len, input_size)

        c_emb = F.dropout(c_emb, self.drop_prob, self.training)
        q_emb = F.dropout(q_emb, self.drop_prob, self.training)

        p_i = self.passage_rnn(c_emb, c_len)    # (batch_size, c_len, h)
        q = self.query_rnn(q_emb, q_len)        # (batch_size, 1, h)
        q_t = q.permute(0, 2, 1)                # (batch_size, h, 1)

        logits_start = self.att_start(p_i, q_t)
        logits_end = self.att_end(p_i, q_t)

        probs_start = masked_softmax(logits_start, c_mask, log_softmax=True)   # (batch_size, c_len)
        probs_end = masked_softmax(logits_end, c_mask, log_softmax=True)       # (batch_size, c_len)

        return probs_start, probs_end
