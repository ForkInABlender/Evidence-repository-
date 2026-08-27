# Dylan Kenneth Eliot


import numpy as np
from pybrain3.structure import FeedForwardNetwork, LinearLayer, FullConnection, Module
from pybrain3.structure.modules.neuronlayer import NeuronLayer
from dim3_neuronlayer import Dim3NeuronLayer, TRAINING, clip_grad

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

class CustomBatchLayer(Module):
    """Custom layer to handle batch input with _forwardImplementation and _backwardImplementation."""

    def __init__(self, dim_in=None, dim_out=None, dim=None):
        if dim_in is None: dim_in = dim
        if dim_out is None: dim_out = dim
        super(CustomBatchLayer, self).__init__(dim_in, dim_out)
        self.argdict = {'dim_in': dim_in, 'dim_out': dim_out}
        self.name = "CustomBatchLayer"

    def _forwardImplementation(self, inbuf, outbuf):
        batch_size = inbuf.size // self.indim
        input_matrix = np.reshape(inbuf, (batch_size, self.indim))
        activated_matrix = 1 / (1 + np.exp(-np.clip(input_matrix, -700, 700)))  # Sigmoid with clipping
        np.copyto(outbuf, activated_matrix.flatten())

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        batch_size = inbuf.size // self.indim
        outbuf_matrix = np.reshape(outbuf, (batch_size, self.indim))
        sigmoid_grad = outbuf_matrix * (1 - outbuf_matrix)
        np.copyto(inerr, outerr * sigmoid_grad.flatten())

class EmbeddingLayer(LinearLayer):
    def __init__(self, vocab_size=None, embedding_dim=None, dim=None):
        # pybrain3 XML reader reconstructs with dim=<outdim>
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

        # Concatenate attention heads and pass through the final linear layer
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


# Constants
VOCAB_SIZE = 50257
D_MODEL = 16
FFN_DIM = 16
NUM_HEADS = 4
NUM_BLOCKS = 12

# Create the GPT-like network
net = FeedForwardNetwork()
inLayer = LinearLayer(VOCAB_SIZE); inLayer.name = 'in'
net.addInputModule(inLayer)

# Add embedding layer and custom batch layer
embedding = EmbeddingLayer(VOCAB_SIZE, D_MODEL); embedding.name = 'embedding'
batch_layer = CustomBatchLayer(D_MODEL, D_MODEL); batch_layer.name = 'batch'
net.addModule(embedding)
net.addModule(batch_layer)
net.addConnection(FullConnection(inLayer, embedding))
net.addConnection(FullConnection(embedding, batch_layer))

# Add multi-head attention and transformer blocks
prev_layer = batch_layer
for i in range(NUM_BLOCKS):
    norm1 = LayerNorm(D_MODEL); norm1.name = f'norm1_{i}'
    attention = MultiHeadSelfAttention(D_MODEL, D_MODEL, NUM_HEADS); attention.name = f'attn_{i}'
    dim3_layer = Dim3NeuronLayer(
        in_dim=D_MODEL, out_dim=D_MODEL, embed_dropout=0.1, attn_dropout=0.1,
        activation_function='gelu', lr=0.0001, weight_decay=0.01, gradient_clipping=1.0
    ); dim3_layer.name = f'dim3_{i}'
    norm2 = LayerNorm(D_MODEL); norm2.name = f'norm2_{i}'
    
    net.addModule(norm1)
    net.addModule(attention)
    net.addModule(dim3_layer)
    net.addModule(norm2)

    net.addConnection(FullConnection(prev_layer, norm1))
    net.addConnection(FullConnection(norm1, attention))
    net.addConnection(FullConnection(attention, dim3_layer))
    net.addConnection(FullConnection(dim3_layer, norm2))
    
    prev_layer = norm2

# Add final output layer
out_layer = LinearLayer(VOCAB_SIZE)
net.addOutputModule(out_layer)
net.addConnection(FullConnection(prev_layer, out_layer))

# Finalize the network
net.sortModules()




#from main__GPT_Model import NetworkWriter
#from main__GPT_Model import NetworkReader


#NetworkWriter.writeToFile(net, 'gpt_3d.xml')
#loaded_net = NetworkReader.readFrom('gpt_3d.xml')
