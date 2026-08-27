# Dylan Kenneth Eliot & GPT-4-Plugins (Beta Edition)

"""
This is mostly for parts to use in modeling an actual AI. But with the bits for emulating the brain.

https://chat.openai.com/share/2a9f0d74-5687-4eb8-a4ed-8195b8270516
"""
from pybrain3.structure import SoftmaxLayer, LinearLayer
from pybrain3.datasets import SupervisedDataSet
from pybrain3.structure import FeedForwardNetwork, FullConnection
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.structure.modules.neuronlayer import NeuronLayer
import numpy as np


class FeedForwardLayer(NeuronLayer):
    def __init__(self, indim, outdim):
        super().__init__(indim, outdim)
        self.argdict = {'indim': indim, 'outdim': outdim}
        self.weights = np.random.randn(indim, outdim)
        self.bias = np.random.randn(outdim)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = inbuf @ self.weights + self.bias

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        self.weights -= inbuf[:, np.newaxis] @ outerr[np.newaxis, :]
        self.bias -= outerr
        inerr[:] = outerr @ self.weights.T


class EmbeddingLayer(LinearLayer):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__(embedding_dim)
        self.argdict = {'vocab_size': vocab_size, 'embedding_dim': embedding_dim}
        self.name = 'EmbeddingLayer'
        self.embeddings = np.random.randn(vocab_size, embedding_dim)

    def _forwardImplementation(self, inbuf, outbuf):
        # Fix: < instead of <= to avoid out-of-bounds access
        idx = np.argmax(inbuf)
        self.token_idx = idx if idx < len(self.embeddings) else len(self.embeddings) - 1
        outbuf[:] = self.embeddings[self.token_idx]

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        self.embeddings[self.token_idx] -= outerr
        inerr[:] = 0


class GeLULayer(NeuronLayer):
    def __init__(self, dim):
        super().__init__(dim, dim)
        self.argdict = {'dim': dim}

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = 0.5 * inbuf * (1 + np.tanh(np.sqrt(2 / np.pi) * (inbuf + 0.044715 * inbuf ** 3)))

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        # Fix: use clipped x consistently for both cdf and pd
        x = np.clip(inbuf, -10, 10)
        inner = np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
        cdf = 0.5 * (1 + np.tanh(inner))
        pd = np.sqrt(2 / np.pi) * (1 + 0.134145 * x ** 2) * (1 / np.cosh(inner)) ** 2
        inerr[:] = outerr * (cdf + x * pd)


class AttentionLayer(NeuronLayer):
    def __init__(self, indim, outdim):
        super().__init__(indim, outdim)
        self.argdict = {'indim': indim, 'outdim': outdim}
        self.attention_weights = np.random.rand(indim, outdim)

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = np.dot(inbuf, self.attention_weights)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        gradient = np.dot(inbuf.T, outerr)
        self.attention_weights -= gradient
        inerr[:] = np.dot(outerr, self.attention_weights.T)


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


class MultiHeadSelfAttention(NeuronLayer):
    def __init__(self, indim, outdim, num_heads):
        super().__init__(indim, outdim)
        self.argdict = {'indim': indim, 'outdim': outdim, 'num_heads': num_heads}
        self.num_heads = num_heads
        self.depth = indim // num_heads
        self.W_q = np.random.rand(indim, indim)
        self.W_k = np.random.rand(indim, indim)
        self.W_v = np.random.rand(indim, indim)
        self.W_o = np.random.rand(indim, outdim)

    def _forwardImplementation(self, inbuf, outbuf):
        if inbuf.ndim == 1:
            inbuf = inbuf[np.newaxis, :]
        self._inbuf = inbuf
        Q = np.dot(inbuf, self.W_q)
        K = np.dot(inbuf, self.W_k)
        V = np.dot(inbuf, self.W_v)
        self.Q = np.split(Q, self.num_heads, axis=1)
        self.K = np.split(K, self.num_heads, axis=1)
        self.V = np.split(V, self.num_heads, axis=1)

        self.attention_weights = []
        heads = []
        for i in range(self.num_heads):
            aw = softmax(np.dot(self.Q[i], self.K[i].T) / np.sqrt(self.depth), axis=-1)
            self.attention_weights.append(aw)
            heads.append(np.dot(aw, self.V[i]))

        self.concatenated_heads = np.concatenate(heads, axis=1)
        # Fix: flatten result to match 1D outbuf
        outbuf[:] = np.dot(self.concatenated_heads, self.W_o).ravel()[:outbuf.size]

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        seq_len = self._inbuf.shape[0]
        outerr_2d = outerr.reshape(seq_len, -1)
        d_concat = np.dot(outerr_2d, self.W_o.T)
        d_heads = np.split(d_concat, self.num_heads, axis=1)
        dQ, dK, dV = [], [], []
        for i in range(self.num_heads):
            d_out = d_heads[i]
            dV.append(np.dot(self.attention_weights[i].T, d_out))
            d_aw = np.dot(d_out, self.V[i].T)
            dQK = d_aw * self.attention_weights[i] * (1 - self.attention_weights[i])
            dQ.append(np.dot(dQK, self.K[i]))
            dK.append(np.dot(dQK, self.Q[i]))
        dQ_total = np.concatenate(dQ, axis=1)
        dK_total = np.concatenate(dK, axis=1)
        dV_total = np.concatenate(dV, axis=1)
        dinput = np.dot(dQ_total, self.W_q.T) + np.dot(dK_total, self.W_k.T) + np.dot(dV_total, self.W_v.T)
        inerr[:] = dinput.ravel()[:inerr.size]
        self.W_q -= np.dot(self._inbuf.T, dQ_total)
        self.W_k -= np.dot(self._inbuf.T, dK_total)
        self.W_v -= np.dot(self._inbuf.T, dV_total)
        self.W_o -= np.dot(self.concatenated_heads.T, outerr_2d)


class LayerNorm(NeuronLayer):
    def __init__(self, size, eps=1e-6):
        super().__init__(size, size)
        self.argdict = {'size': size, 'eps': eps}
        self.gamma = np.ones(size)
        self.beta = np.zeros(size)
        self.eps = eps

    def _forwardImplementation(self, inbuf, outbuf):
        mean = np.mean(inbuf)
        std = np.std(inbuf)
        outbuf[:] = self.gamma * (inbuf - mean) / (std + self.eps) + self.beta

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        N = inbuf.size
        outerr = np.clip(outerr, -1e6, 1e6)
        mean = np.mean(inbuf)
        std = max(np.std(inbuf), self.eps)
        xhat = (inbuf - mean) / std
        # Fix: cache repeated np.sum calls
        sum_outerr = np.sum(outerr)
        sum_outerr_xhat = np.sum(outerr * xhat)
        self.beta -= sum_outerr
        self.gamma -= sum_outerr_xhat
        inerr[:] = (self.gamma / (N * std)) * (N * outerr - sum_outerr - xhat * sum_outerr_xhat)


VOCAB_SIZE = 50257  # GPT-3/4 use 50257; use 120 for quick local testing
D_MODEL = 16
NUM_BLOCKS = 8      # 94-96 for GPT-3/4; 4 for local testing
NUM_HEADS = 16
FFN_DIM = 16


def build_network():
    net = FeedForwardNetwork()
    inLayer = LinearLayer(VOCAB_SIZE); inLayer.name = 'in'
    net.addInputModule(inLayer)
    embedding = EmbeddingLayer(VOCAB_SIZE, D_MODEL); embedding.name = 'embedding'
    net.addModule(embedding)
    net.addConnection(FullConnection(inLayer, embedding))
    attention = MultiHeadSelfAttention(D_MODEL, D_MODEL, NUM_HEADS); attention.name = 'attention'
    net.addModule(attention)
    net.addConnection(FullConnection(embedding, attention))
    prev_layer = attention
    for i in range(NUM_BLOCKS):
        norm1 = LayerNorm(D_MODEL); norm1.name = f'norm1_{i}'
        net.addModule(norm1)
        net.addConnection(FullConnection(prev_layer, norm1))
        ffn1 = LinearLayer(D_MODEL, FFN_DIM); ffn1.name = f'ffn1_{i}'
        net.addModule(ffn1)
        net.addConnection(FullConnection(norm1, ffn1))
        gelu = GeLULayer(FFN_DIM); gelu.name = f'gelu_{i}'
        net.addModule(gelu)
        net.addConnection(FullConnection(ffn1, gelu))
        ffn2 = LinearLayer(FFN_DIM, D_MODEL); ffn2.name = f'ffn2_{i}'
        net.addModule(ffn2)
        net.addConnection(FullConnection(gelu, ffn2))
        norm2 = LayerNorm(D_MODEL); norm2.name = f'norm2_{i}'
        net.addModule(norm2)
        net.addConnection(FullConnection(ffn2, norm2))
        prev_layer = norm2
    outLayer = SoftmaxLayer(VOCAB_SIZE); outLayer.name = 'out'
    net.addOutputModule(outLayer)
    net.addConnection(FullConnection(prev_layer, outLayer))
    net.sortModules()
    return net



import sys
import pybrain3.tools.xml.networkreader as _nr
from pybrain3.tools.xml.networkwriter import NetworkWriter
from pybrain3.tools.xml.networkreader import NetworkReader

# Patch readBuildable to resolve any module-qualified class (e.g. __main__.X or mymodule.X)
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

net = build_network()
NetworkWriter.writeToFile(net, 'my_model.xml')
loaded_net = NetworkReader.readFrom('my_model.xml')
print("reloaded")
from main_gpt3_3d import net as gpt3d
print("saved 3d")
NetworkWriter.writeToFile(gpt3d, 'my_model_gpt3d.xml')
loaded_net = NetworkReader.readFrom('my_model_gpt3d.xml')
print("loaded 3d")
