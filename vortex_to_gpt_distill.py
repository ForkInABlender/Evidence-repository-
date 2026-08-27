# Dylan Kenneth Eliot
"""
Distill from vortex_distill.py LSTM teacher -> main_gpt3_3d.py GPT student.

Uses n-gram soft targets (matching vortex_distill.py's teacher pattern) over
GPT's 50257 token space. Vortex math encodes positional weight on the input
one-hot — no extra dimensions, 50257 I/O fully preserved.
"""

import numpy as np
from collections import defaultdict, Counter
from pybrain3.tools.xml import NetworkReader, NetworkWriter
from pybrain3.supervised.trainers import BackpropTrainer

from vortex_distill import (
    neural_generate, CORPUS,
    char2id, id2char, id2idx, idx2id, char2cluster, vocab_size,
    digital_root, family_group, vortex_pos,
    NGramTeacher
)

GPT_IO = 50257   # fixed net I/O size
NGRAM_N = 4
EPOCHS  = 10

# ── load vortex LSTM teacher ───────────────────────────────────────────────────
lstm_teacher = NetworkReader.readFrom("vortex_model.xml")
print("Loaded vortex LSTM teacher.")

# ── one corpus sample: exactly 50257 chars ─────────────────────────────────────
sample = neural_generate(
    lstm_teacher, "the ",
    char2id, id2char, id2idx, idx2id,
    vocab_size, char2cluster,
    length=GPT_IO, temp=0.6
)[:GPT_IO]
print(f"Sample length: {len(sample)} chars")

# ── word vocab from sample ─────────────────────────────────────────────────────
words  = sample.split()
unique = sorted(set(words))
w2i    = {w: i for i, w in enumerate(unique)}
i2w    = {i: w for w, i in w2i.items()}
V      = len(unique)   # actual vocab size (small)
print(f"Word vocab: {V}")

# ── n-gram teacher ─────────────────────────────────────────────────────────────
ngram = NGramTeacher(n=NGRAM_N)
ngram.train(sample)
sorted_chars = sorted(char2id.keys())
c2i = {c: i for i, c in enumerate(sorted_chars)}

def vortex_weight(pos):
    dr = digital_root(pos)
    vp = vortex_pos(dr)
    fg = family_group(pos)
    w  = (vp + 1) / 6.0 if vp >= 0 else 0.5
    return max(w * (1.0 - fg * 0.1), 0.1)

def encode_token(token_id, pos):
    """Sparse-friendly: one-hot over V (actual vocab), scaled by vortex weight."""
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

# ── load GPT student ───────────────────────────────────────────────────────────
from main_gpt3_3d import net
student = net

# ── train one sample at a time (no dataset object) ────────────────────────────
tokens = [w2i[w] for w in words if w in w2i]
print(f"Tokens: {len(tokens)}  training {EPOCHS} epochs...")

for epoch in range(1, EPOCHS + 1):
    total_err = 0.0
    for i in range(len(tokens) - 1):
        x_small = encode_token(tokens[i], i + 1)
        ctx     = [i2w[t] for t in tokens[max(0, i - NGRAM_N + 1):i + 1] if t in i2w]
        y_small = soft_target(ctx, tokens[i + 1])

        # pad/project into GPT_IO space
        x = np.zeros(GPT_IO); x[:V] = x_small
        y = np.zeros(GPT_IO); y[:V] = y_small

        out   = np.array(student.activate(x), dtype=np.float64)
        err   = out - y
        total_err += float(np.dot(err, err))
        student.backActivate(err)
    print(f"  epoch {epoch:3d}  mse={total_err / max(len(tokens)-1, 1):.6f}")

# ── save ───────────────────────────────────────────────────────────────────────
NetworkWriter.writeToFile(student, "gpt_distilled.xml")
print("Saved gpt_distilled.xml")
student = NetworkReader.readFrom("gpt_distilled.xml")
# ── Test ───────────────────────────────────────────────────────────────────────
print("\n── GPT distilled responses ──────────────────────────────────────")

def gpt_generate(net, seed_words, length=20, temp=0.8):
    result = list(seed_words)
    for pos in range(len(seed_words), len(seed_words) + length):
        last = result[-1]
        tid  = w2i.get(last, 0)
        x    = np.zeros(GPT_IO); x[:V] = encode_token(tid, pos)
        probs = np.array(net.activate(x), dtype=np.float64)[:V]
        log_p = np.log(np.clip(probs, 1e-9, 1.0)) / temp
        log_p -= log_p.max()
        probs  = np.exp(log_p); probs /= probs.sum()
        next_id = np.random.choice(V, p=probs)
        result.append(i2w.get(next_id, "<unk>"))
    return ' '.join(result)

for seed in [["the", "cat"], ["language", "is"], ["to", "be"]]:
    # filter seed to known vocab
    seed = [w for w in seed if w in w2i] or [words[0]]
    print(f"\n[seed='{' '.join(seed)}']")
    print(gpt_generate(student, seed, length=20, temp=0.8))
