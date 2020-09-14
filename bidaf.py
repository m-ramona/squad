import torch
import torch.nn as nn
import torch.nn.functional as F

from mylayers import AttentionLayer, RNNLayer, AttentionFlowLayer, BiDAFOutputLayer, EmbeddingLayer


class BiDAF(nn.Module):
    def __init__(self, word_vectors, hidden_size,
                 char_vectors=None,
                 drop_prob=0.,
                 context_layers=1,
                 use_gru=True,
                 mod_layers=2,
                 share_rnn=False,
                 highway=False):
        super(BiDAF, self).__init__()

        self.hidden_size = hidden_size
        self.drop_prob = drop_prob

        self.embed = EmbeddingLayer(word_vectors,
                                    hidden_size,
                                    char_vectors=char_vectors,
                                    drop_prob=drop_prob,
                                    highway=highway)

        self.context_rnn = RNNLayer(input_size=self.hidden_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=context_layers,
                                    drop_prob=drop_prob,
                                    end_of_seq=False,
                                    use_gru=use_gru)

        if share_rnn:
            self.query_rnn = self.context_rnn
        else:
            self.query_rnn = RNNLayer(input_size=self.hidden_size,
                                      hidden_size=self.hidden_size,
                                      num_layers=context_layers,
                                      drop_prob=drop_prob,
                                      end_of_seq=False,
                                      use_gru=use_gru)

        self.flow = AttentionFlowLayer(2*self.hidden_size,
                                       drop_prob=drop_prob)

        self.modeling = RNNLayer(input_size=8*self.hidden_size,
                                 hidden_size=self.hidden_size,
                                 num_layers=mod_layers,
                                 drop_prob=drop_prob,
                                 end_of_seq=False,
                                 use_gru=use_gru)

        self.output = BiDAFOutputLayer(self.hidden_size,
                                       drop_prob=drop_prob,
                                       use_gru=use_gru)

    def forward(self, cw_idxs, qw_idxs, cc_idxs=None, qc_idxs=None):
        # cw_idxs, cc_idxs (batch_size, c_len)
        # qw_idxs, qc_idxs (batch_size, q_len)

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.embed(cw_idxs, cc_idxs)    # (batch_size, c_len, input_size)
        q_emb = self.embed(qw_idxs, qc_idxs)    # (batch_size, q_len, input_size)

        # c_emb = F.dropout(c_emb, self.drop_prob, self.training)
        # q_emb = F.dropout(q_emb, self.drop_prob, self.training)

        h = self.context_rnn(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        u = self.query_rnn(q_emb, q_len)        # (batch_size, q_len, 2 * hidden_size)

        g = self.flow(h, u, c_mask, q_mask)
        m = self.modeling(g, c_len)

        p1, p2 = self.output(g, m, c_mask)

        return p1, p2
