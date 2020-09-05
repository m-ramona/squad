from unittest import TestCase

import torch

from attentive_reader import RNNLayer, AttentiveReaderModel, AttentionLayer, AttentiveReaderOutput

batch = 3
clen = 6
qlen = 5
input_size = 2
hidden_size = 5
emb_size = 4
vocab = 13


class TestAttentionLayer(TestCase):
    def test_forward(self):
        layer = AttentionLayer(hidden_size)
        p_i = torch.Tensor(batch, clen, hidden_size)
        q_t = torch.Tensor(batch, hidden_size, 1)
        logits = layer(p_i, q_t)
        self.assertEquals(logits.size(), (batch, clen))


class TestAttentiveReaderOutput(TestCase):
    def test_forward(self):
        logits = torch.rand((batch, clen))
        mask_gen = torch.empty(logits.size()).fill_(0.5)
        mask = torch.bernoulli(mask_gen)
        layer = AttentiveReaderOutput(hidden_size)
        probs = layer(logits, mask, log_softmax=False)
        self.assertEqual(probs.size(), logits.size())
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones((batch,))))
        for prob_seq, mask_seq in zip(probs, mask):
            masked_probs = prob_seq[mask_seq == 0.]
            zeros = torch.zeros((sum(mask_seq == 0),))
            self.assertTrue(torch.allclose(masked_probs, zeros))


class TestRNNLayer(TestCase):
    def get_input(self):
        x = torch.rand((batch, clen, input_size))
        lengths = torch.randint(1, clen, (batch, ))
        for x_seq, seq_len in zip(x, lengths):
            x_seq[seq_len:, :] = 0
        return x, lengths

    def test_forward_context(self):
        x, lengths = self.get_input()
        for use_gru in [False, True]:
            layer = RNNLayer(input_size, hidden_size, end_of_seq=False, use_gru=use_gru)
            out = layer(x, lengths)
            self.assertEqual(out.size(), (batch, clen, 2*hidden_size))
            one = torch.ByteTensor([1]).squeeze()
            for out_seq, seq_len in zip(out, lengths):
                all_zero = (out_seq[seq_len:, :] == 0.).all()
                self.assertTrue(all_zero.equal(one))

    def test_forward_query(self):
        x, lengths = self.get_input()

        for use_gru in [False, True]:
            layer = RNNLayer(input_size, hidden_size, end_of_seq=True, use_gru=use_gru)
            out = layer(x, lengths)
            self.assertEqual(out.size(), (batch, 1, 2*hidden_size))


class TestAttentiveReaderModel(TestCase):
    def get_idxs(self, batch, seq_len, vocab_size):
        lengths = torch.randint(1, seq_len, (batch, ))
        idxs = torch.randint(1, vocab_size, (batch, seq_len))
        for seq, seq_len in zip(idxs, lengths):
            seq[seq_len:] = 0
        return idxs, lengths

    def test_forward(self):
        word_vectors = torch.rand((vocab, emb_size))
        word_vectors[0, :] = 0.
        cw_idxs, c_lengths = self.get_idxs(batch, clen, vocab)
        qw_idxs, q_lengths = self.get_idxs(batch, qlen, vocab)
        layer = AttentiveReaderModel(word_vectors, hidden_size)
        probs_start, probs_end = layer(cw_idxs, qw_idxs)
        self.assertEqual(probs_start.size(), (batch, clen))
        self.assertEqual(probs_end.size(), (batch, clen))
        for probs_start_seq, probs_end_seq, clen_seq in zip(probs_start, probs_end, c_lengths):
            zeros = torch.zeros((clen-clen_seq,))
            self.assertTrue(torch.allclose(torch.exp(probs_start_seq[clen_seq:]), zeros))

