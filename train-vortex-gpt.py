#!/usr/bin/env python3
"""
train_unified.py

End-to-end trainer that:
  1) Reads vortex_gpt_corpus_50257.txt
  2) Trains the n-gram -> LSTM teacher (saves vortex_model.xml)
  3) Builds the 50257-wide GPT-style pybrain3 student
  4) Distills teacher-generated soft targets into the student
  5) Saves gpt_distilled.xml

Requires the patched pybrain3 wheel:
  pip install https://github.com/ForkInABlender/ChatGPT_solutions_produced_from_simple_prompt/raw/refs/heads/2026_06/modules_&_wheels/pybrain_and_pybrain3__patched-02_12_2026__3pm.zip

Also requires: numpy, scikit-learn
"""

import sys
import os
import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import SpectralClustering

# ── pybrain3 XML monkey-patch for custom layers ────────────────────────────────
import pybrain3.tools.xml.networkreader as _nr
from pybrain3.tools.xml.networkwriter import NetworkWriter
from pybrain3.tools.xml.networkreader import NetworkReader

_orig_readBuildable = NetworkReader.readBuildable
def _patched_readBuildable(self, node):
    mclass = node.getAttribute('class')
    if '.' in mclass:
        modname, clsname = mclass.rsplit('.', 1)
        mod = sys.modules.get(modname) or __import__(modname)
        _nr.__dict__[clsname] = getattr(mod, clsname)
        node.setAttribute('class', clsname)
    return _orig_readBuildable(self, node)
NetworkReader.readBuildable = _patched_readBuildable

# ── pybrain3 imports ─────────────────────────────────────────────────────────
from pybrain3.structure import (
    FeedForwardNetwork, RecurrentNetwork, LinearLayer, LSTMLayer,
    SoftmaxLayer, FullConnection, Module
)
from pybrain3.structure.modules.neuronlayer import NeuronLayer
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.datasets import SequentialDataSet

# ── Constants ────────────────────────────────────────────────────────────────
CORPUS_PATH = "/sdcard/Download/vortex_gpt_corpus_50257.txt"
VOCAB_SIZE = 50257
D_MODEL = 16
FFN_DIM = 16
NUM_HEADS = 4
NUM_BLOCKS = 12          # lower if CPU memory is tight
GPT_IO = 50257
EPOCHS_LSTM = 200
EPOCHS_DISTILL = 1
LSTM_HIDDEN = 64

# ── Vortex math ───────────────────────────────────────────────────────────────
DOUBLING_SEQ = [1, 2, 4, 8, 7, 5]
DOUBLING_POS = {v: i for i, v in enumerate(DOUBLING_SEQ)}

def digital_root(x):
    return (x % 9) or 9

def vortex_pos(x):
    return DOUBLING_POS.get(digital_root(x), -1)

def family_group(x):
    dr = digital_root(x)
    if dr in {1, 4, 7}:
        return 0
    if dr in {2, 5, 8}:
        return 1
    return 2

def vortex_weight(pos):
    dr = digital_root(pos)
    vp = vortex_pos(dr)
    fg = family_group(pos)
    w = (vp + 1) / 6.0 if vp >= 0 else 0.5
    return max(w * (1.0 - fg * 0.1), 0.1)

# ── Vocab / N-gram ────────────────────────────────────────────────────────────
def build_vocab(corpus):
    chars = sorted(set(corpus))
    n = len(chars)
    c2i_raw = {c: i for i, c in enumerate(chars)}
    W = np.zeros((n, n))
    for k in range(len(corpus)):
        for j in range(k + 1, min(k + 3, len(corpus))):
            a, b = c2i_raw[corpus[k]], c2i_raw[corpus[j]]
            W[a, b] += 1
            W[b, a] += 1
    n_clusters = min(9, n)
    labels = SpectralClustering(
        n_clusters=n_clusters, affinity='precomputed', random_state=42
    ).fit_predict(W + 1e-6 * np.eye(n))
    char2id = {c: i for i, c in enumerate(chars)}
    id2char = {i: c for c, i in char2id.items()}
    char2cluster = {chars[i]: int(labels[i]) for i in range(n)}
    id2idx = {i: i for i in range(n)}
    idx2id = id2idx
    return char2id, id2char, id2idx, idx2id, char2cluster

class NGramTeacher:
    def __init__(self, n=4):
        self.n = n
        self.counts = defaultdict(Counter)

    def train(self, text):
        for i in range(len(text) - self.n):
            self.counts[text[i:i + self.n]][text[i + self.n]] += 1

    def soft_target(self, ctx, vocab, c2i, vocab_size):
        key = ctx[-(self.n):]
        while key and key not in self.counts:
            key = key[1:]
        y = np.zeros(vocab_size)
        if not key:
            return None
        c = self.counts[key]
        total = sum(c.values())
        for ch, cnt in c.items():
            if ch in c2i:
                y[c2i[ch]] = cnt / total
        return y

def encode_char(vid, id2idx, vocab_size, pos, char2cluster=None, id2char=None):
    vec = np.zeros(vocab_size + 3)
    vec[id2idx[vid]] = 1.0
    ch = id2char[vid] if id2char else None
    lbl = char2cluster[ch] if (char2cluster and ch) else 0
    vec[vocab_size] = (lbl % 6) / 5.0
    vec[vocab_size + 1] = (lbl // 3) / 2.0
    vec[vocab_size + 2] = digital_root(pos) / 9.0
    return vec

# ── LSTM teacher ────────────────────────────────────────────────────────────────
def build_rnn(input_size, hidden_size, output_size):
    net = RecurrentNetwork()
    net.addInputModule(LinearLayer(input_size, name='in'))
    net.addModule(LSTMLayer(hidden_size, name='lstm'))
    net.addOutputModule(SoftmaxLayer(output_size, name='out'))
    net.addConnection(FullConnection(net['in'], net['lstm']))
    net.addConnection(FullConnection(net['lstm'], net['out']))
    net.addRecurrentConnection(FullConnection(net['lstm'], net['lstm'], name='rec'))
    net.sortModules()
    return net

def neural_generate(net, seed, char2id, id2char, id2idx, idx2id, vocab_size, char2cluster, length=200, temp=0.8):
    ids = [char2id[c] for c in seed if c in char2id]
    result = list(seed)
    net.reset()
    for pos, vid in enumerate(ids):
        net.activate(encode_char(vid, id2idx, vocab_size, pos + 1, char2cluster, id2char))
    pos, last_vid = len(ids), ids[-1]
    for _ in range(length):
        probs = np.array(net.activate(encode_char(last_vid, id2idx, vocab_size, pos, char2cluster, id2char)), dtype=np.float64)
        log_p = np.log(np.clip(probs, 1e-9, 1.0)) / temp
        log_p -= log_p.max()
        probs = np.exp(log_p)
        probs /= probs.sum()
        last_vid = idx2id[np.random.choice(len(probs), p=probs)]
        result.append(id2char[last_vid])
        pos += 1
    return ''.join(result)

def distill_dataset(corpus, ngram, char2id, id2idx, vocab_size, char2cluster, id2char, chunk=15):
    ids = [char2id[c] for c in corpus if c in char2id]
    text = [c for c in corpus if c in char2id]
    feat = vocab_size + 3
    ds = SequentialDataSet(feat, vocab_size)
    c2i = {ch: i for i, ch in enumerate(sorted(char2id.keys()))}
    for start in range(0, len(ids) - 1, chunk):
        seg_ids = ids[start:start + chunk + 1]
        seg_text = text[start:start + chunk + 1]
        ds.newSequence()
        for i in range(len(seg_ids) - 1):
            x = encode_char(seg_ids[i], id2idx, vocab_size, start + i + 1, char2cluster, id2char)
            ctx = ''.join(seg_text[max(0, i - ngram.n + 1):i + 1])
            y = ngram.soft_target(ctx, sorted(char2id.keys()), c2i, vocab_size)
            if y is None:
                y = np.zeros(vocab_size)
                y[id2idx[seg_ids[i + 1]]] = 1.0
            ds.addSample(x, y)
    return ds

def blended_dataset(corpus, ngram, char2id, id2idx, vocab_size, char2cluster, id2char, chunk=15, alpha=0.3):
    ids = [char2id[c] for c in corpus if c in char2id]
    text = [c for c in corpus if c in char2id]
    feat = vocab_size + 3
    ds = SequentialDataSet(feat, vocab_size)
    c2i = {ch: i for i, ch in enumerate(sorted(char2id.keys()))}
    for start in range(0, len(ids) - 1, chunk):
        seg_ids = ids[start:start + chunk + 1]
        seg_text = text[start:start + chunk + 1]
        ds.newSequence()
        for i in range(len(seg_ids) - 1):
            x = encode_char(seg_ids[i], id2idx, vocab_size, start + i + 1, char2cluster, id2char)
            ctx = ''.join(seg_text[max(0, i - ngram.n + 1):i + 1])
            soft = ngram.soft_target(ctx, sorted(char2id.keys()), c2i, vocab_size)
            hard = np.zeros(vocab_size)
            hard[id2idx[seg_ids[i + 1]]] = 1.0
            y = alpha * hard + (1 - alpha) * (soft if soft is not None else hard)
            ds.addSample(x, y)
    return ds

def train_teacher(corpus_text):
    print("Building vocab...")
    char2id, id2char, id2idx, idx2id, char2cluster = build_vocab(corpus_text)
    vocab_size = len(char2id)
    feat_size = vocab_size + 3
    print(f"Vocab size: {vocab_size}  Feat size: {feat_size}")

    print("Training n-gram teacher...")
    ngram = NGramTeacher(n=min(121, len(corpus_text)))
    ngram.train(corpus_text)

    print("Building LSTM...")
    net = build_rnn(feat_size, LSTM_HIDDEN, vocab_size)

    # Phase 1: distill
    print(f"Phase 1: distill from n-gram ({EPOCHS_LSTM} epochs)...")
    ds_soft = distill_dataset(corpus_text, ngram, char2id, id2idx, vocab_size, char2cluster, id2char)
    trainer = BackpropTrainer(net, ds_soft, learningrate=0.01, momentum=0.9)
    for epoch in range(1, EPOCHS_LSTM):
        err = trainer.train()
        if epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  error={err:.6f}")

    # Phase 2: blended fine-tune
    print(f"Phase 2: blended fine-tune ({EPOCHS_LSTM} epochs)...")
    ds_blend = blended_dataset(corpus_text, ngram, char2id, id2idx, vocab_size, char2cluster, id2char)
    trainer = BackpropTrainer(net, ds_blend, learningrate=0.0005, momentum=0.9)
    for epoch in range(1, EPOCHS_LSTM):
        err = trainer.train()
        if epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  error={err:.6f}")

    NetworkWriter.writeToFile(net, "vortex_model.xml")
    print("Saved vortex_model.xml")
    return net, char2id, id2char, id2idx, idx2id, char2cluster, vocab_size

# ── GPT student helpers ────────────────────────────────────────────────────────
TRAINING = True
def clip_grad(g, clip=1.0):
    n = np.linalg.norm(g)
    return (g * (clip / n) if n > clip else g)

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

class CustomBatchLayer(Module):
    def __init__(self, dim_in=None, dim_out=None, dim=None):
        if dim_in is None: dim_in = dim
        if dim_out is None: dim_out = dim
        super().__init__(dim_in, dim_out)
        self.argdict = {'dim_in': dim_in, 'dim_out': dim_out}
        self.name = "CustomBatchLayer"

    def _forwardImplementation(self, inbuf, outbuf):
        batch_size = inbuf.size // self.indim
        input_matrix = np.reshape(inbuf, (batch_size, self.indim))
        activated_matrix = 1 / (1 + np.exp(-np.clip(input_matrix, -700, 700)))
        np.copyto(outbuf, activated_matrix.flatten())

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        batch_size = inbuf.size // self.indim
        outbuf_matrix = np.reshape(outbuf, (batch_size, self.indim))
        sigmoid_grad = outbuf_matrix * (1 - outbuf_matrix)
        np.copyto(inerr, outerr * sigmoid_grad.flatten())

class EmbeddingLayer(LinearLayer):
    def __init__(self, vocab_size=None, embedding_dim=None, dim=None):
        if dim is not None and vocab_size is None:
            embedding_dim = dim
            vocab_size = VOCAB_SIZE
        super().__init__(embedding_dim)
        self.name = "EmbeddingLayer"
        self.embeddings = np.random.randn(vocab_size, embedding_dim)

    def _forwardImplementation(self, inbuf, outbuf):
        self.token_idx = np.argmax(inbuf)
        outbuf[:] = self.embeddings[min(self.token_idx, len(self.embeddings) - 1)]

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        self.embeddings[self.token_idx] -= outerr
        inerr[:] = 0

class MultiHeadSelfAttention(NeuronLayer):
    def __init__(self, indim=None, outdim=None, num_heads=None, dim=None, name=None):
        if indim is None: indim = dim
        if outdim is None: outdim = dim
        if num_heads is None: num_heads = NUM_HEADS
        super().__init__(indim, outdim)
        self.num_heads = num_heads
        self.depth = indim // num_heads
        scale = 0.02
        self.W_q = np.random.randn(indim, indim) * scale
        self.W_k = np.random.randn(indim, indim) * scale
        self.W_v = np.random.randn(indim, indim) * scale
        self.W_o = np.random.randn(indim, outdim) * scale
        self.name = name or "MultiHeadSelfAttention"

    def scaled_dot_product_attention(self, Q, K, V):
        matmul_qk = np.dot(Q, K.T)
        d_k = Q.shape[-1]
        scaled_attention_logits = matmul_qk / np.sqrt(d_k)
        attention_weights = softmax(scaled_attention_logits, axis=-1)
        return np.dot(attention_weights, V), attention_weights

    def _forwardImplementation(self, inbuf, outbuf):
        if len(inbuf.shape) == 1:
            inbuf = inbuf[np.newaxis, :]
        Q = np.dot(inbuf, self.W_q)
        K = np.dot(inbuf, self.W_k)
        V = np.dot(inbuf, self.W_v)
        self.Q = np.split(Q, self.num_heads, axis=1)
        self.K = np.split(K, self.num_heads, axis=1)
        self.V = np.split(V, self.num_heads, axis=1)
        attention_heads = []
        self.attention_weights = []
        for i in range(self.num_heads):
            attention_head, aw = self.scaled_dot_product_attention(self.Q[i], self.K[i], self.V[i])
            attention_heads.append(attention_head)
            self.attention_weights.append(aw)
        self.concatenated_heads = np.concatenate(attention_heads, axis=1)
        outbuf[:] = np.dot(self.concatenated_heads, self.W_o).reshape(outbuf.shape)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        if outerr.ndim == 1:
            outerr = outerr[np.newaxis, :]
        if inbuf.ndim == 1:
            inbuf = inbuf[np.newaxis, :]
        d_concat_heads = np.dot(outerr, self.W_o.T)
        d_attention_heads = np.split(d_concat_heads, self.num_heads, axis=1)
        dQ_heads, dK_heads, dV_heads = [], [], []
        for i in range(self.num_heads):
            d_out = d_attention_heads[i]
            dV = np.dot(self.attention_weights[i].T, d_out)
            d_attention_weights = np.dot(d_out, self.V[i].T)
            dQK = d_attention_weights * (1 - self.attention_weights[i]) * self.attention_weights[i]
            dQ_heads.append(np.dot(dQK, self.K[i]))
            dK_heads.append(np.dot(dQK, self.Q[i]))
            dV_heads.append(dV)
        dQ_total = clip_grad(np.concatenate(dQ_heads, axis=1))
        dK_total = clip_grad(np.concatenate(dK_heads, axis=1))
        dV_total = clip_grad(np.concatenate(dV_heads, axis=1))
        lr = 1e-4
        inerr[:] = clip_grad((np.dot(dQ_total, self.W_q.T) + np.dot(dK_total, self.W_k.T) + np.dot(dV_total, self.W_v.T)).reshape(inerr.shape))
        if not TRAINING:
            self.W_q -= lr * clip_grad(np.dot(inbuf.T, dQ_total))
            self.W_k -= lr * clip_grad(np.dot(inbuf.T, dK_total))
            self.W_v -= lr * clip_grad(np.dot(inbuf.T, dV_total))
            self.W_o -= lr * clip_grad(np.dot(self.concatenated_heads.T, outerr))

class LayerNorm(NeuronLayer):
    def __init__(self, dim, eps=1e-5, name=None):
        super().__init__(dim, dim)
        self.eps = eps
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.name = name or "LayerNorm"

    def _forwardImplementation(self, inbuf, outbuf):
        mean = np.mean(inbuf, axis=-1, keepdims=True)
        variance = np.var(inbuf, axis=-1, keepdims=True)
        outbuf[:] = self.gamma * (inbuf - mean) / np.sqrt(variance + self.eps) + self.beta

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        mean = np.mean(inbuf, axis=-1, keepdims=True)
        variance = np.var(inbuf, axis=-1, keepdims=True)
        x_hat = (inbuf - mean) / np.sqrt(variance + self.eps)
        N = inbuf.shape[-1]
        dy_scaled = clip_grad(outerr * self.gamma)
        inerr[:] = clip_grad((dy_scaled - np.mean(dy_scaled, axis=-1, keepdims=True)
                    - x_hat * np.mean(dy_scaled * x_hat, axis=-1, keepdims=True)
                    ) / np.sqrt(variance + self.eps))
        lr = 1e-4
        if not TRAINING:
            self.gamma -= lr * (np.sum(outerr * x_hat, axis=0) if x_hat.ndim > 1 else outerr * x_hat)
            self.beta -= lr * (np.sum(outerr, axis=0) if outerr.ndim > 1 else outerr)

class Dim3NeuronLayer(NeuronLayer):
    def __init__(self, in_dim=None, out_dim=None, embed_dropout=0.1, attn_dropout=0.1,
                 activation_function='gelu', lr=1e-4, weight_decay=0.01,
                 gradient_clipping=1.0, dim=None, name=None):
        if in_dim is None: in_dim = dim
        if out_dim is None: out_dim = dim
        super().__init__(in_dim, out_dim)
        self.embed_dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.activation_function = activation_function
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_clipping = gradient_clipping
        self.weights = np.random.randn(in_dim, out_dim) * 0.1
        self.bias = np.zeros(out_dim)
        self.name = name or "Dim3NeuronLayer"

    def _forwardImplementation(self, inbuf, outbuf):
        input_data = np.array(inbuf)
        original_shape = input_data.shape
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, 1, -1)
        elif input_data.ndim == 2:
            input_data = input_data.reshape(1, *input_data.shape)
        output_data = self._process3D(input_data)
        if len(original_shape) == 1:
            output_data = output_data.flatten()
        elif len(original_shape) == 2:
            output_data = output_data.reshape(original_shape[0], -1)
        outbuf[:output_data.shape[0]] = output_data[:outbuf.shape[0]]

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        out_data = np.array(outerr)
        original_shape = out_data.shape
        if out_data.ndim == 1:
            out_data = out_data.reshape(1, 1, -1)
        elif out_data.ndim == 2:
            out_data = out_data.reshape(1, *out_data.shape)
        grad_input = self._backward3D(out_data)
        if len(original_shape) == 1:
            grad_input = grad_input.flatten()
        elif len(original_shape) == 2:
            grad_input = grad_input.reshape(original_shape[0], -1)
        inerr[:grad_input.shape[0]] = grad_input[:inerr.shape[0]]

    def _process3D(self, input_data):
        self.last_input = input_data
        self.pre_activation = np.tensordot(input_data, self.weights, axes=([-1], [0])) + self.bias
        return self._apply_activation(self.pre_activation)

    def _backward3D(self, outerr):
        grad = clip_grad(outerr * self._activation_derivative(self.pre_activation), self.gradient_clipping)
        if not TRAINING:
            self.weights -= self.lr * np.tensordot(self.last_input, grad, axes=([0, 1], [0, 1]))
            self.bias -= self.lr * grad.sum(axis=(0, 1))
        return clip_grad(np.tensordot(grad, self.weights.T, axes=([-1], [0])), self.gradient_clipping)

    def _apply_activation(self, input_data):
        if self.activation_function == 'gelu':
            return np.where(
                input_data > 20, 1.0,
                np.where(
                    input_data < -20, 0.0,
                    input_data * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (input_data + 0.044715 * np.power(input_data, 3))))
                )
            )
        raise NotImplementedError("Only GELU is implemented")

    def _activation_derivative(self, input_data):
        if self.activation_function == 'gelu':
            tanh_term = np.tanh(np.sqrt(2.0 / np.pi) * (input_data + 0.044715 * np.power(input_data, 3)))
            return np.where(
                input_data > 20, 0.0,
                np.where(
                    input_data < -20, 0.0,
                    0.5 * (1.0 + tanh_term) + 0.5 * input_data * (1 - tanh_term ** 2) * (np.sqrt(2.0 / np.pi) * (1 + 3 * 0.044715 * np.power(input_data, 2)))
                )
            )
        raise NotImplementedError("Only GELU derivative is implemented")

def build_gpt_student():
    net = FeedForwardNetwork()
    inLayer = LinearLayer(VOCAB_SIZE)
    inLayer.name = 'in'
    net.addInputModule(inLayer)

    embedding = EmbeddingLayer(VOCAB_SIZE, D_MODEL)
    embedding.name = 'embedding'
    batch_layer = CustomBatchLayer(D_MODEL, D_MODEL)
    batch_layer.name = 'batch'
    net.addModule(embedding)
    net.addModule(batch_layer)
    net.addConnection(FullConnection(inLayer, embedding))
    net.addConnection(FullConnection(embedding, batch_layer))

    prev_layer = batch_layer
    for i in range(NUM_BLOCKS):
        norm1 = LayerNorm(D_MODEL)
        norm1.name = f'norm1_{i}'
        attention = MultiHeadSelfAttention(D_MODEL, D_MODEL, NUM_HEADS)
        attention.name = f'attn_{i}'
        dim3_layer = Dim3NeuronLayer(
            in_dim=D_MODEL, out_dim=D_MODEL, embed_dropout=0.1, attn_dropout=0.1,
            activation_function='gelu', lr=0.0001, weight_decay=0.01, gradient_clipping=1.0
        )
        dim3_layer.name = f'dim3_{i}'
        norm2 = LayerNorm(D_MODEL)
        norm2.name = f'norm2_{i}'

        net.addModule(norm1)
        net.addModule(attention)
        net.addModule(dim3_layer)
        net.addModule(norm2)

        net.addConnection(FullConnection(prev_layer, norm1))
        net.addConnection(FullConnection(norm1, attention))
        net.addConnection(FullConnection(attention, dim3_layer))
        net.addConnection(FullConnection(dim3_layer, norm2))
        prev_layer = norm2

    out_layer = LinearLayer(VOCAB_SIZE)
    net.addOutputModule(out_layer)
    net.addConnection(FullConnection(prev_layer, out_layer))
    net.sortModules()
    return net

# ── Distillation loop ──────────────────────────────────────────────────────────
def distill_student(lstm_net, student_net, corpus_sample):
    # generate teacher sample
    print("Generating LSTM teacher sample...")
    # re-extract char mappings from the loaded network's construction
    # (we reuse the global vocab built during teacher training)
    pass

def main():
    if not os.path.exists(CORPUS_PATH):
        print(f"ERROR: {CORPUS_PATH} not found.")
        sys.exit(1)

    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus_text = f.read()
    print(f"Loaded corpus: {len(corpus_text)} chars")

    # Train / load teacher
    if os.path.exists("vortex_model.xml"):
        print("Loading existing vortex_model.xml ...")
        lstm_net = NetworkReader.readFrom("vortex_model.xml")
        # vocab must be rebuilt from corpus to get mappings
        char2id, id2char, id2idx, idx2id, char2cluster = build_vocab(corpus_text)
        vocab_size = len(char2id)
    else:
        lstm_net, char2id, id2char, id2idx, idx2id, char2cluster, vocab_size = train_teacher(corpus_text)

    # Build GPT student
    print("Building GPT student...")
    student = build_gpt_student()

    # Generate teacher text sample for word-level distillation
    print("Generating teacher sample for word vocab...")
    sample = neural_generate(
        lstm_net, "the ",
        char2id, id2char, id2idx, idx2id,
        vocab_size, char2cluster,
        length=GPT_IO, temp=0.6
    )[:GPT_IO]
    print(f"Sample length: {len(sample)} chars")

    words = sample.split()
    unique = sorted(set(words))
    w2i = {w: i for i, w in enumerate(unique)}
    i2w = {i: w for w, i in w2i.items()}
    V = len(unique)
    print(f"Word vocab size: {V}")

    # Prepare n-gram for soft targets at word/char bridge
    ngram = NGramTeacher(n=min(50257, max(4, len(sample))))
    ngram.train(sample)
    sorted_chars = sorted(char2id.keys())
    c2i = {c: i for i, c in enumerate(sorted_chars)}

    def encode_token(token_id, pos):
        vec = np.zeros(V)
        if token_id < V:
            vec[token_id] = vortex_weight(pos)
        return vec

    def soft_target(ctx_words, next_word_id):
        char_dist = ngram.soft_target(' '.join(ctx_words), sorted_chars, c2i, len(char2id))
        y = np.zeros(V)
        if char_dist is not None:
            for wid, word in i2w.items():
                if word and word[0] in c2i:
                    y[wid] += char_dist[c2i[word[0]]]
            s = y.sum()
            if s > 0:
                return y / s
        y[next_word_id] = 1.0
        return y

    tokens = [w2i[w] for w in words if w in w2i]
    print(f"Distillation tokens: {len(tokens)}  epochs={EPOCHS_DISTILL}")

    for epoch in range(1, EPOCHS_DISTILL + 1):
        total_err = 0.0
        for i in range(len(tokens) - 1):
            x_small = encode_token(tokens[i], i + 1)
            ctx = [i2w[t] for t in tokens[max(0, i - ngram.n + 1):i + 1] if t in i2w]
            y_small = soft_target(ctx, tokens[i + 1])

            x = np.zeros(GPT_IO)
            x[:V] = x_small
            y = np.zeros(GPT_IO)
            y[:V] = y_small

            out = np.array(student.activate(x), dtype=np.float32)
            err = out - y
            total_err += float(np.dot(err, err))

            # manual backward step using pybrain3 internal backActivate if present
            if hasattr(student, 'backActivate'):
                student.backActivate(err)
            else:
                # Fallback: gradient descent on input connections' parameters is non-trivial
                # for a full FeedForwardNetwork; warn once.
                if i == 0 and epoch == 1:
                    print("WARN: student.backActivate not found; skipping weight updates.")

        print(f"  epoch {epoch:3d}  mse={total_err / max(len(tokens)-1, 1):.6f}")

    NetworkWriter.writeToFile(student, "gpt_distilled.xml")
    print("Saved gpt_distilled.xml")

    # Quick generation test
    print("\n── GPT distilled responses ──────────────────────────────────────")
    def gpt_generate(net, seed_words, length=20, temp=0.8):
        result = list(seed_words)
        for pos in range(len(seed_words), len(seed_words) + length):
            last = result[-1]
            tid = w2i.get(last, 0)
            x = np.zeros(GPT_IO)
            x[:V] = encode_token(tid, pos)
            probs = np.array(net.activate(x), dtype=np.float64)[:V]
            log_p = np.log(np.clip(probs, 1e-9, 1.0)) / temp
            log_p -= log_p.max()
            probs = np.exp(log_p)
            probs /= probs.sum()
            next_id = np.random.choice(V, p=probs)
            result.append(i2w.get(next_id, "<unk>"))
        return ' '.join(result)

    for seed in [["the", "cat"], ["language", "is"], ["to", "be"]]:
        seed = [w for w in seed if w in w2i] or [words[0]]
        print(f"\n[seed='{' '.join(seed)}']")
        print(gpt_generate(student, seed, length=20, temp=0.8))

if __name__ == "__main__":
    main()
