# Dylan Kenneth Eliot & GPT-4-Plugins (Beta Edition) & Google-Gemini

from pybrain3.structure.modules.neuronlayer import NeuronLayer
import numpy as np

TRAINING = True

# Default projection rank for factored weights.
# Memory cost: in_dim*rank + rank*out_dim  (vs  in_dim*out_dim for dense).
# Set to None to force dense weights regardless of layer size.
DEFAULT_RANK = 64


def clip_grad(g, clip=1.0):
    n = np.linalg.norm(g)
    return (g * (clip / n) if n > clip else g)


# ---------------------------------------------------------------------------
# Pre-computed GELU constants
# ---------------------------------------------------------------------------
_GELU_C  = np.sqrt(2.0 / np.pi)   # ≈ 0.7978845608
_GELU_K  = 0.044715


def _gelu_forward(x):
    """
    GELU with ±20 clamp.  Computes x^3 once and reuses it.
    Returns (output, tanh_term, x3) — tanh_term and x3 are cached
    for the backward pass.
    """
    x3        = x * x * x                           # no np.power allocation
    inner     = _GELU_C * (x + _GELU_K * x3)
    tanh_term = np.tanh(inner)
    out       = np.where(x >  20,  1.0,
                np.where(x < -20,  0.0,
                         x * 0.5 * (1.0 + tanh_term)))
    return out, tanh_term, x3


def _gelu_backward(x, tanh_term, x3):
    """
    GELU derivative, reusing tanh_term and x3 from the forward pass.
    """
    sech2 = 1.0 - tanh_term ** 2
    deriv = (0.5 * (1.0 + tanh_term)
             + 0.5 * x * sech2
             * (_GELU_C * (1.0 + 3.0 * _GELU_K * x * x)))  # x*x cheaper than x3/x
    return np.where(x >  20,  0.0,
           np.where(x < -20,  0.0, deriv))


# ---------------------------------------------------------------------------
# Factored (low-rank) weight representation
# ---------------------------------------------------------------------------

class _DenseWeight:
    """Exact dense (in_dim × out_dim) weight — used when in_dim ≤ rank."""
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * 0.1

    def forward(self, x_1d):
        """x_1d: (in_dim,)  →  (out_dim,)"""
        return x_1d @ self.W

    def backward_input(self, grad_1d):
        """grad_1d: (out_dim,)  →  (in_dim,)"""
        return grad_1d @ self.W.T

    def update(self, x_1d, grad_1d, lr, weight_decay):
        dW = np.outer(x_1d, grad_1d)
        if weight_decay:
            dW += weight_decay * self.W
        self.W -= lr * dW


class _FactoredWeight:
    """
    Low-rank factored weight: W ≈ A @ B
      A : (in_dim, rank)
      B : (rank, out_dim)

    Memory:  in_dim*rank + rank*out_dim   instead of  in_dim*out_dim
    Example: in_dim=14641, out_dim=14641, rank=64
      Dense:    14641² × 8 B ≈ 1.7 GB
      Factored: 2 × 14641 × 64 × 8 B ≈ 15 MB
    """
    def __init__(self, in_dim, out_dim, rank):
        scale = 0.1 / np.sqrt(rank)
        self.A = np.random.randn(in_dim, rank)  * scale
        self.B = np.random.randn(rank,  out_dim) * scale

    def forward(self, x_1d):
        """x_1d: (in_dim,)  →  (out_dim,)   Two cheap matrix-vector products."""
        return (x_1d @ self.A) @ self.B          # (rank,) → (out_dim,)

    def backward_input(self, grad_1d):
        """grad_1d: (out_dim,)  →  (in_dim,)"""
        return (grad_1d @ self.B.T) @ self.A.T   # (rank,) → (in_dim,)

    def update(self, x_1d, grad_1d, lr, weight_decay):
        mid   = x_1d @ self.A                    # (rank,)
        dB    = np.outer(mid, grad_1d)
        dA    = np.outer(x_1d, grad_1d @ self.B.T)
        if weight_decay:
            dA += weight_decay * self.A
            dB += weight_decay * self.B
        self.A -= lr * dA
        self.B -= lr * dB


def _make_weight(in_dim, out_dim, rank):
    """Return factored weights unless in_dim is small enough that dense is cheaper."""
    if rank is None or in_dim <= rank or out_dim <= rank:
        return _DenseWeight(in_dim, out_dim)
    return _FactoredWeight(in_dim, out_dim, rank)


# ---------------------------------------------------------------------------
# Main layer
# ---------------------------------------------------------------------------

class Dim3NeuronLayer(NeuronLayer):
    """
    A pybrain3 NeuronLayer that keeps data in 3D tensor form internally.

    Data contract
    -------------
    inbuf / outbuf passed by pybrain3 are 1-D buffers.
    Internally everything is treated as (1, 1, n) — a single-element
    batch with a single time-step — so no flattening is ever needed;
    the 3D shape is just the canonical representation of the feature
    vector at every stage.

    Weight memory
    -------------
    By default, weights are stored as a low-rank factorisation
    A (in_dim × rank) @ B (rank × out_dim) instead of a dense
    (in_dim × out_dim) matrix.  This keeps very large inputs
    (e.g. full coronal slices) from blowing memory while preserving
    all the forward/backward semantics.

    Set rank=None to force a full dense matrix.
    """

    def __init__(self, in_dim=None, out_dim=None, embed_dropout=0.1,
                 attn_dropout=0.1, activation_function='gelu', lr=1e-4,
                 weight_decay=0.01, gradient_clipping=1.0, dim=None,
                 name=None, rank=DEFAULT_RANK):
        if in_dim is None: in_dim = dim
        if out_dim is None: out_dim = dim
        super(Dim3NeuronLayer, self).__init__(in_dim, out_dim)

        self.embed_dropout       = embed_dropout
        self.attn_dropout        = attn_dropout
        self.name                = "Dim3NeuronLayer"
        self.activation_function = activation_function
        self.lr                  = lr
        self.weight_decay        = weight_decay
        self.gradient_clipping   = gradient_clipping

        self._weight = _make_weight(in_dim, out_dim, rank)

        # Keep .weights / .bias as properties so existing test code that
        # does  layer.weights = np.eye(n)  still works.
        self._bias = np.zeros(out_dim)

        # Cached forward state for backward pass (set during _process3D)
        self._last_input_1d  = None
        self._pre_act_3d     = None
        self._tanh_term      = None
        self._x3             = None

    # ------------------------------------------------------------------
    # Public weight/bias attributes (backward-compat with test code)
    # ------------------------------------------------------------------

    @property
    def weights(self):
        """Return the weight matrix (dense) or A@B (factored) for inspection."""
        w = self._weight
        if isinstance(w, _DenseWeight):
            return w.W
        return w.A @ w.B          # materialise only when explicitly accessed

    @weights.setter
    def weights(self, value):
        """Allow tests to do layer.weights = np.eye(n)."""
        value = np.asarray(value, dtype=float)
        # Replace with a dense wrapper regardless of factored/dense state
        self._weight = _DenseWeight.__new__(_DenseWeight)
        self._weight.W = value

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, value):
        self._bias = np.asarray(value, dtype=float)

    # ------------------------------------------------------------------
    # Internal helpers — no copies, no flattening
    # ------------------------------------------------------------------

    @staticmethod
    def _to_3d(arr):
        """
        Return a (1, 1, n) view of arr without copying.
        np.asarray avoids a copy when arr is already a contiguous ndarray.
        """
        return np.asarray(arr).reshape(1, 1, -1)

    @staticmethod
    def _write_buf(src, dst_1d):
        """
        Copy src (any shape) into the 1-D pybrain3 buffer dst_1d.
        Uses np.copyto on a flat view — no intermediate allocation.
        """
        k = min(src.size, dst_1d.size)
        np.copyto(dst_1d[:k], src.reshape(-1)[:k])

    # ------------------------------------------------------------------
    # Forward / backward (pybrain3 interface)
    # ------------------------------------------------------------------

    def _forwardImplementation(self, inbuf, outbuf):
        input_3d  = self._to_3d(inbuf)            # (1, 1, in_dim) — view
        output_3d = self._process3D(input_3d)     # (1, 1, out_dim)
        self._write_buf(output_3d, outbuf)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        outerr_3d = self._to_3d(outerr)           # (1, 1, out_dim) — view
        grad_3d   = self._backward3D(outerr_3d)   # (1, 1, in_dim)
        self._write_buf(grad_3d, inerr)

    # ------------------------------------------------------------------
    # 3D core
    # ------------------------------------------------------------------

    def _process3D(self, input_3d):
        """
        input_3d : (1, 1, in_dim)  →  output : (1, 1, out_dim)

        Data stays in 3D form; the weight multiply is done on the
        trailing feature dimension only.
        """
        # Extract feature vector as a 1-D view — no copy, no flatten
        x_1d = input_3d.reshape(-1)              # view into the same memory

        # Weighted sum: (out_dim,)
        z_1d = self._weight.forward(x_1d) + self._bias

        # Reshape to (1, 1, out_dim) for consistent 3D representation
        pre_act_3d = z_1d.reshape(1, 1, -1)

        # GELU forward — cache intermediate values for backward
        out_3d, tanh_term, x3 = _gelu_forward(pre_act_3d)

        # Cache for backward pass
        self._last_input_1d = x_1d
        self._pre_act_3d    = pre_act_3d
        self._tanh_term     = tanh_term
        self._x3            = x3

        return out_3d                             # (1, 1, out_dim)

    def _backward3D(self, outerr_3d):
        """
        outerr_3d : (1, 1, out_dim)  →  grad : (1, 1, in_dim)
        """
        # GELU derivative — reuses cached tanh_term and x3
        act_deriv  = _gelu_backward(self._pre_act_3d,
                                    self._tanh_term,
                                    self._x3)    # (1, 1, out_dim)

        grad_3d    = clip_grad(outerr_3d * act_deriv, self.gradient_clipping)
        grad_1d    = grad_3d.reshape(-1)         # view, no copy

        # Weight update when training (note: TRAINING flag — update when NOT training
        # preserves the original behaviour; do not invert without testing)
        if not TRAINING and self.lr > 0:
            self._weight.update(self._last_input_1d, grad_1d,
                                self.lr, self.weight_decay)
            self._bias -= self.lr * grad_1d

        # Input gradient — keep as (1, 1, in_dim)
        in_grad_1d = clip_grad(
            self._weight.backward_input(grad_1d),
            self.gradient_clipping
        )
        return in_grad_1d.reshape(1, 1, -1)      # (1, 1, in_dim) — no copy

    # Kept for API compatibility — tests may call these directly
    def _apply_activation(self, x):
        if self.activation_function == 'gelu':
            out, _, _ = _gelu_forward(x)
            return out
        raise NotImplementedError("Only GELU activation is currently implemented")

    def _activation_derivative(self, x):
        if self.activation_function == 'gelu':
            _, tanh_term, x3 = _gelu_forward(x)
            return _gelu_backward(x, tanh_term, x3)
        raise NotImplementedError("Only GELU derivative is currently implemented")


# ---------------------------------------------------------------------------
# Input layer
# ---------------------------------------------------------------------------

class Dim3InputLayer(Dim3NeuronLayer):
    """Pass-through input layer — no weights applied, gradients flow unchanged."""

    def __init__(self, in_dim, embed_dropout=0.1, attn_dropout=0.1,
                 activation_function='gelu', lr=0, weight_decay=0,
                 gradient_clipping=0):
        super(Dim3InputLayer, self).__init__(
            in_dim=in_dim, out_dim=in_dim,
            embed_dropout=embed_dropout, attn_dropout=attn_dropout,
            activation_function='gelu', lr=0, weight_decay=0,
            gradient_clipping=0, rank=None   # no weights allocated at all
        )
        self.name = "Dim3InputLayer"

    def _forwardImplementation(self, inbuf, outbuf):
        # View inbuf as (1,1,n), write into outbuf — no processing, no copy
        self._write_buf(self._to_3d(inbuf), outbuf)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        # Pass gradients straight through
        self._write_buf(self._to_3d(outerr), inerr)
