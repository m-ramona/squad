from unittest import TestCase

import torch

from attentive_reader import AttentiveReaderModel
from bidaf import BiDAF
from mylayers import AttentionLayer, RNNLayer, AttentionFlowLayer, BiDAFOutputLayer
from util import masked_softmax

batch = 3
clen = 6
qlen = 5
input_size = 2
hidden_size = 5
emb_size = 4
vocab = 13


def get_rnn_input(batch, seq_len, input_size, lengths=None):
    if lengths is None:
        lengths = torch.randint(1, seq_len, (batch,))
    x = torch.rand((batch, seq_len, input_size))
    masks = torch.ones((batch, seq_len), dtype=torch.uint8)

    for x_seq, mask, seq_len in zip(x, masks, lengths):
        x_seq[seq_len:, :] = 0
        mask[seq_len:] = 0
    return x, lengths, masks


def get_idxs(batch, seq_len, vocab_size):
    lengths = torch.randint(1, seq_len, (batch, ))
    idxs = torch.randint(1, vocab_size, (batch, seq_len))
    for seq, seq_len in zip(idxs, lengths):
        seq[seq_len:] = 0
    return idxs, lengths


def get_word_vectors(vocab_size, emb_size):
    word_vectors = torch.rand((vocab_size, emb_size))
    word_vectors[0, :] = 0.
    return word_vectors


class TestAttentionLayer(TestCase):
    def test_forward(self):
        layer = AttentionLayer(hidden_size)
        p_i = torch.Tensor(batch, clen, hidden_size)
        q_t = torch.Tensor(batch, hidden_size, 1)
        logits = layer(p_i, q_t)
        self.assertEquals(logits.size(), (batch, clen))


class TestMaskedSoftmax(TestCase):
    def test_forward(self):
        logits = torch.rand((batch, clen))
        mask_gen = torch.empty(logits.size()).fill_(0.5)
        mask = torch.bernoulli(mask_gen)

        probs = masked_softmax(logits, mask, log_softmax=False)
        self.assertEqual(probs.size(), logits.size())
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones((batch,))))
        for prob_seq, mask_seq in zip(probs, mask):
            masked_probs = prob_seq[mask_seq == 0.]
            zeros = torch.zeros((sum(mask_seq == 0),))
            self.assertTrue(torch.allclose(masked_probs, zeros))


class TestRNNLayer(TestCase):
    def get_input(self):
        return get_rnn_input(batch, clen, input_size)[:-1]

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
    def test_forward(self):
        word_vectors = get_word_vectors(vocab, emb_size)
        cw_idxs, c_lengths = get_idxs(batch, clen, vocab)
        qw_idxs, q_lengths = get_idxs(batch, qlen, vocab)

        model = AttentiveReaderModel(word_vectors, hidden_size)
        probs_start, probs_end = model(cw_idxs, qw_idxs)
        self.assertEqual(probs_start.size(), (batch, clen))
        self.assertEqual(probs_end.size(), (batch, clen))
        for probs_start_seq, probs_end_seq, clen_seq in zip(probs_start, probs_end, c_lengths):
            zeros = torch.zeros((clen-clen_seq,))
            self.assertTrue(torch.allclose(torch.exp(probs_start_seq[clen_seq:]), zeros))


class TestAttentionFlowLayer(TestCase):
    def get_input(self):
        h, h_lengths, h_mask = get_rnn_input(batch, clen, hidden_size*2)
        u, u_lengths, u_mask = get_rnn_input(batch, qlen, hidden_size*2)
        return h, u, h_mask, u_mask

    def test_similarity_matrix(self):
        h, u, h_mask, u_mask = self.get_input()
        layer = AttentionFlowLayer(hidden_size*2)
        S = layer.similarity_matrix(h, u)
        self.assertEqual(S.size(), (batch, clen, qlen))

    def test_attention_vectors(self):
        h, u, h_mask, u_mask = self.get_input()
        layer = AttentionFlowLayer(hidden_size*2)
        H_tilde, U_tilde = layer.attention_vectors(h, u, h_mask, u_mask)
        self.assertEqual(H_tilde.size(), h.size())
        self.assertEqual(U_tilde.size(), h.size())

    def test_G(self):
        h, u, h_mask, u_mask = self.get_input()
        layer = AttentionFlowLayer(hidden_size*2)
        H_tilde, U_tilde = layer.attention_vectors(h, u, h_mask, u_mask)
        G = layer.G(h, H_tilde, U_tilde)
        self.assertEqual(G.size(), (batch, clen, hidden_size*8))

    def test_forward(self):
        h, u, h_mask, u_mask = self.get_input()
        layer = AttentionFlowLayer(hidden_size*2)
        out = layer(h, u, h_mask, u_mask)
        self.assertEqual(out.size(), (batch, clen, hidden_size*8))


class TestBiDAFOutputLayer(TestCase):
    def test_forward(self):
        g, lengths, masks = get_rnn_input(batch, clen, 8*hidden_size)
        m, _, _ = get_rnn_input(batch, clen, 2*hidden_size, lengths)

        layer = BiDAFOutputLayer(hidden_size, use_gru=True)
        p1, p2 = layer(g, m, masks)
        self.assertEqual(p1.size(), (batch, clen))
        self.assertEqual(p2.size(), (batch, clen))
        self.assertTrue(torch.allclose(p1.exp().sum(-1), torch.ones((batch,))))
        self.assertTrue(torch.allclose(p2.exp().sum(-1), torch.ones((batch,))))
        self.assertTrue(torch.equal(p1 > -1e29, masks))
        self.assertTrue(torch.equal(p2 > -1e29, masks))


class TestBiDAF(TestCase):
    def test_forward(self):
        word_vectors = get_word_vectors(vocab, emb_size)
        cw_idxs, c_lengths = get_idxs(batch, clen, vocab)
        qw_idxs, q_lengths = get_idxs(batch, qlen, vocab)

        model = BiDAF(word_vectors, hidden_size)
        p1, p2 = model(cw_idxs, qw_idxs)
        self.assertEqual(p1.size(), (batch, clen))
        self.assertEqual(p2.size(), (batch, clen))
        self.assertTrue(torch.allclose(p1.exp().sum(-1), torch.ones((batch,))))
        self.assertTrue(torch.allclose(p2.exp().sum(-1), torch.ones((batch,))))
