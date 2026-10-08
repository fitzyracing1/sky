#!/usr/bin/env python3
"""nano-llm: a tiny decoder-only transformer language model (NumPy only).

Character-level GPT. Trains with Adam, samples with temperature, saves weights
as .npz. No PyTorch. This is a real next-token model, not a Markov toy.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = ROOT / "nano_llm.npz"
DEFAULT_CORPUS = ROOT / "corpus.txt"

CORPUS = """\
An LLM is a language model. It reads tokens and predicts the next token.
A decoder-only transformer does this with attention over earlier tokens.
Attention lets each position look back at the context it already saw.
The model never looks ahead. That is the causal mask.
Embeddings turn characters into vectors. Position embeddings mark order.
Each block has layer norm, multi-head attention, a residual, then a small MLP.
The head maps the last hidden state to a score for every character.
Training minimizes cross-entropy between those scores and the true next character.
Sampling draws the next character from the softened scores, then feeds it back.
Small models ramble. Larger models and more data make the ramble sharper.
This nano LLM lives in NumPy. It has no framework and no GPU requirement.
Repeat the loop: embed, attend, mix, project, sample, append, attend again.
Language is compression. Prediction is the training signal. Generation is the use.
"""


def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


def encode(text: str, stoi: dict) -> np.ndarray:
    return np.array([stoi[c] for c in text], dtype=np.int64)


def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * inv
    y = g * xhat + b
    cache = (xhat, inv, g)
    return y, cache


def layer_norm_backward(dy, cache):
    xhat, inv, g = cache
    n = dy.shape[-1]
    dxhat = dy * g
    dvar = np.sum(dxhat * xhat, axis=-1, keepdims=True) * (-0.5) * (inv ** 3)
    dmu = np.sum(dxhat, axis=-1, keepdims=True) * (-inv) + dvar * np.mean(-2.0 * xhat / inv, axis=-1, keepdims=True)
    # simpler standard LN backward:
    # dx = (1/n) * inv * (n*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
    dx = (inv / n) * (
        n * dxhat
        - np.sum(dxhat, axis=-1, keepdims=True)
        - xhat * np.sum(dxhat * xhat, axis=-1, keepdims=True)
    )
    dg = np.sum(dy * xhat, axis=tuple(range(dy.ndim - 1)))
    db = np.sum(dy, axis=tuple(range(dy.ndim - 1)))
    return dx, dg, db


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def causal_mask(t: int) -> np.ndarray:
    return np.tril(np.ones((t, t), dtype=np.float64))


class NanoLM:
    def __init__(self, vocab: int, n_embd=32, n_head=4, n_layer=2, block=48, seed=7):
        assert n_embd % n_head == 0
        self.vocab = vocab
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block = block
        self.head_dim = n_embd // n_head
        rng = np.random.default_rng(seed)
        s = 0.02
        self.wte = rng.normal(0, s, (vocab, n_embd))
        self.wpe = rng.normal(0, s, (block, n_embd))
        self.layers = []
        for _ in range(n_layer):
            self.layers.append({
                "ln1_g": np.ones(n_embd),
                "ln1_b": np.zeros(n_embd),
                "wq": rng.normal(0, s, (n_embd, n_embd)),
                "wk": rng.normal(0, s, (n_embd, n_embd)),
                "wv": rng.normal(0, s, (n_embd, n_embd)),
                "wo": rng.normal(0, s, (n_embd, n_embd)),
                "ln2_g": np.ones(n_embd),
                "ln2_b": np.zeros(n_embd),
                "fc1": rng.normal(0, s, (n_embd, 4 * n_embd)),
                "fc1_b": np.zeros(4 * n_embd),
                "fc2": rng.normal(0, s, (4 * n_embd, n_embd)),
                "fc2_b": np.zeros(n_embd),
            })
        self.lnf_g = np.ones(n_embd)
        self.lnf_b = np.zeros(n_embd)
        self.lm_head = rng.normal(0, s, (n_embd, vocab))
        self.lm_bias = np.zeros(vocab)
        self._init_adam()

    def _named(self):
        params = {"wte": self.wte, "wpe": self.wpe, "lnf_g": self.lnf_g, "lnf_b": self.lnf_b,
                  "lm_head": self.lm_head, "lm_bias": self.lm_bias}
        for i, layer in enumerate(self.layers):
            for k, v in layer.items():
                params[f"L{i}.{k}"] = v
        return params

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self._named().items()}
        self.v = {k: np.zeros_like(v) for k, v in self._named().items()}
        self.step = 0

    def _split_heads(self, x):
        b, t, c = x.shape
        h, d = self.n_head, self.head_dim
        return x.reshape(b, t, h, d).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        b, h, t, d = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, t, h * d)

    def forward(self, idx):
        b, t = idx.shape
        x = self.wte[idx] + self.wpe[:t]
        caches = []
        mask = causal_mask(t)
        scale = 1.0 / math.sqrt(self.head_dim)
        for layer in self.layers:
            x1, ln1_cache = layer_norm(x, layer["ln1_g"], layer["ln1_b"])
            q = self._split_heads(x1 @ layer["wq"])
            k = self._split_heads(x1 @ layer["wk"])
            v = self._split_heads(x1 @ layer["wv"])
            scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
            scores = np.where(mask[None, None, :, :] > 0, scores, -1e9)
            probs = softmax(scores)
            att = np.matmul(probs, v)
            merged = self._merge_heads(att)
            ao = merged @ layer["wo"]
            x = x + ao
            x2, ln2_cache = layer_norm(x, layer["ln2_g"], layer["ln2_b"])
            h = x2 @ layer["fc1"] + layer["fc1_b"]
            h_act = np.maximum(h, 0.0)
            mo = h_act @ layer["fc2"] + layer["fc2_b"]
            x = x + mo
            caches.append({
                "x_in": None,
                "ln1": ln1_cache,
                "x1": x1,
                "q": q, "k": k, "v": v,
                "probs": probs,
                "merged": merged,
                "x_mid": None,
                "ln2": ln2_cache,
                "x2": x2,
                "h": h,
                "h_act": h_act,
                "scale": scale,
            })
            # store residuals inputs by recomputing from saved tensors later via explicit save
            caches[-1]["res1_in_shape"] = True
        # We need residual inputs. Recompute them by storing explicitly.
        return x, caches

    def forward_train(self, idx, targets):
        """Full forward that also keeps residual inputs for backward."""
        b, t = idx.shape
        tok = self.wte[idx]
        pos = self.wpe[:t]
        x = tok + pos
        caches = []
        mask = causal_mask(t)
        scale = 1.0 / math.sqrt(self.head_dim)
        for layer in self.layers:
            res1 = x
            x1, ln1_cache = layer_norm(x, layer["ln1_g"], layer["ln1_b"])
            q_lin = x1 @ layer["wq"]
            k_lin = x1 @ layer["wk"]
            v_lin = x1 @ layer["wv"]
            q = self._split_heads(q_lin)
            k = self._split_heads(k_lin)
            v = self._split_heads(v_lin)
            scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
            scores = np.where(mask[None, None, :, :] > 0, scores, -1e9)
            probs = softmax(scores)
            att = np.matmul(probs, v)
            merged = self._merge_heads(att)
            ao = merged @ layer["wo"]
            x = res1 + ao
            res2 = x
            x2, ln2_cache = layer_norm(x, layer["ln2_g"], layer["ln2_b"])
            h = x2 @ layer["fc1"] + layer["fc1_b"]
            h_act = np.maximum(h, 0.0)
            mo = h_act @ layer["fc2"] + layer["fc2_b"]
            x = res2 + mo
            caches.append({
                "res1": res1, "x1": x1, "ln1": ln1_cache,
                "q": q, "k": k, "v": v, "probs": probs, "merged": merged,
                "scale": scale, "mask": mask,
                "res2": res2, "x2": x2, "ln2": ln2_cache,
                "h_act": h_act,
            })
        xf, lnf_cache = layer_norm(x, self.lnf_g, self.lnf_b)
        logits = xf @ self.lm_head + self.lm_bias
        logp = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(logp)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        n = targets.size
        gathered = probs[np.arange(b)[:, None], np.arange(t)[None, :], targets]
        loss = -np.log(np.clip(gathered, 1e-12, 1.0)).mean()
        cache = {
            "idx": idx, "tok": tok, "xf": xf, "lnf": lnf_cache,
            "probs": probs, "targets": targets, "layers": caches, "b": b, "t": t,
        }
        return logits, loss, cache

    def backward(self, cache):
        b, t = cache["b"], cache["t"]
        targets = cache["targets"]
        probs = cache["probs"]
        dlogits = probs.copy()
        dlogits[np.arange(b)[:, None], np.arange(t)[None, :], targets] -= 1.0
        dlogits /= (b * t)
        xf = cache["xf"]
        d_lm_head = np.einsum("btc,btv->cv", xf, dlogits)
        d_lm_bias = dlogits.sum(axis=(0, 1))
        dxf = dlogits @ self.lm_head.T
        dx, d_lnf_g, d_lnf_b = layer_norm_backward(dxf, cache["lnf"])
        grads_layers = []
        for li in reversed(range(self.n_layer)):
            layer = self.layers[li]
            c = cache["layers"][li]
            # MLP residual
            dmo = dx
            dres2 = dx
            dh_act = dmo @ layer["fc2"].T
            d_fc2 = np.einsum("bth,btc->hc", c["h_act"], dmo)
            d_fc2_b = dmo.sum(axis=(0, 1))
            dh = dh_act * (c["h_act"] > 0)
            d_fc1 = np.einsum("btc,bth->ch", c["x2"], dh)
            d_fc1_b = dh.sum(axis=(0, 1))
            dx2 = dh @ layer["fc1"].T
            dx_ln2, d_ln2_g, d_ln2_b = layer_norm_backward(dx2, c["ln2"])
            dx = dres2 + dx_ln2
            # Attention residual
            dao = dx
            dres1 = dx
            dmerged = dao @ layer["wo"].T
            d_wo = np.einsum("btc,bto->co", c["merged"], dao)
            datt = dmerged.reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
            dprobs = np.matmul(datt, c["v"].transpose(0, 1, 3, 2))
            dv = np.matmul(c["probs"].transpose(0, 1, 3, 2), datt)
            # softmax backward
            s = c["probs"]
            dscores = s * (dprobs - np.sum(dprobs * s, axis=-1, keepdims=True))
            dscores = dscores * c["scale"]
            # causal positions already zeroed via -1e9; gradient there is ~0 from softmax
            dq = np.matmul(dscores, c["k"])
            dk = np.matmul(dscores.transpose(0, 1, 3, 2), c["q"])
            dq_lin = self._merge_heads(dq)
            dk_lin = self._merge_heads(dk)
            dv_lin = self._merge_heads(dv)
            x1 = c["x1"]
            d_wq = np.einsum("btc,bto->co", x1, dq_lin)
            d_wk = np.einsum("btc,bto->co", x1, dk_lin)
            d_wv = np.einsum("btc,bto->co", x1, dv_lin)
            dx1 = dq_lin @ layer["wq"].T + dk_lin @ layer["wk"].T + dv_lin @ layer["wv"].T
            dx_ln1, d_ln1_g, d_ln1_b = layer_norm_backward(dx1, c["ln1"])
            dx = dres1 + dx_ln1
            grads_layers.append({
                "ln1_g": d_ln1_g, "ln1_b": d_ln1_b,
                "wq": d_wq, "wk": d_wk, "wv": d_wv, "wo": d_wo,
                "ln2_g": d_ln2_g, "ln2_b": d_ln2_b,
                "fc1": d_fc1, "fc1_b": d_fc1_b,
                "fc2": d_fc2, "fc2_b": d_fc2_b,
            })
        grads_layers.reverse()
        # embedding grads
        d_wte = np.zeros_like(self.wte)
        np.add.at(d_wte, cache["idx"].reshape(-1), dx.reshape(-1, self.n_embd))
        d_wpe = dx.sum(axis=0)
        grads = {
            "wte": d_wte, "wpe": d_wpe,
            "lnf_g": d_lnf_g, "lnf_b": d_lnf_b,
            "lm_head": d_lm_head, "lm_bias": d_lm_bias,
        }
        for i, g in enumerate(grads_layers):
            for k, v in g.items():
                grads[f"L{i}.{k}"] = v
        return grads

    def adam(self, grads, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.step += 1
        params = self._named()
        for k, p in params.items():
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            mhat = self.m[k] / (1 - b1 ** self.step)
            vhat = self.v[k] / (1 - b2 ** self.step)
            p -= lr * mhat / (np.sqrt(vhat) + eps)

    def save(self, path: Path, stoi: dict):
        path = Path(path)
        payload = {k: v for k, v in self._named().items()}
        payload["itos"] = np.array([ord(ch) for ch, _ in sorted(stoi.items(), key=lambda kv: kv[1])], dtype=np.int32)
        payload["meta"] = np.array([self.vocab, self.n_embd, self.n_head, self.n_layer, self.block, self.step], dtype=np.int32)
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Path):
        data = np.load(path, allow_pickle=False)
        vocab, n_embd, n_head, n_layer, block, step = data["meta"].tolist()
        model = cls(vocab, n_embd, n_head, n_layer, block)
        named = model._named()
        for k in named:
            named[k][:] = data[k]
        model.step = int(step)
        chars = [chr(int(c)) for c in data["itos"].tolist()]
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return model, stoi, itos


def batches(data: np.ndarray, block: int, batch: int, rng: np.random.Generator):
    n = len(data) - block - 1
    ix = rng.integers(0, max(n, 1), size=batch)
    x = np.stack([data[i:i + block] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


def train(steps=400, lr=3e-3, batch=16, block=48, seed=7):
    text = CORPUS
    if DEFAULT_CORPUS.exists():
        text = DEFAULT_CORPUS.read_text()
    else:
        DEFAULT_CORPUS.write_text(text)
    chars, stoi, itos = build_vocab(text)
    data = encode(text, stoi)
    model = NanoLM(len(chars), n_embd=32, n_head=4, n_layer=2, block=block, seed=seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    last = None
    for step in range(1, steps + 1):
        x, y = batches(data, block, batch, rng)
        _, loss, cache = model.forward_train(x, y)
        grads = model.backward(cache)
        model.adam(grads, lr=lr)
        last = float(loss)
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"step {step:4d}  loss {last:.4f}  ppl {math.exp(min(last, 20)):.2f}")
    model.save(DEFAULT_CKPT, stoi)
    print(f"saved {DEFAULT_CKPT} in {time.time() - t0:.1f}s  vocab={len(chars)}")
    return model, stoi, itos, last


def sample(model: NanoLM, stoi, itos, prompt: str, n=180, temperature=0.8, seed=1):
    rng = np.random.default_rng(seed)
    idx = [stoi.get(c, 0) for c in prompt]
    if not idx:
        idx = [0]
    for _ in range(n):
        window = np.array(idx[-model.block:], dtype=np.int64)[None, :]
        # inference forward
        b, t = window.shape
        x = model.wte[window] + model.wpe[:t]
        mask = causal_mask(t)
        scale = 1.0 / math.sqrt(model.head_dim)
        for layer in model.layers:
            x1, _ = layer_norm(x, layer["ln1_g"], layer["ln1_b"])
            q = model._split_heads(x1 @ layer["wq"])
            k = model._split_heads(x1 @ layer["wk"])
            v = model._split_heads(x1 @ layer["wv"])
            scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
            scores = np.where(mask[None, None, :, :] > 0, scores, -1e9)
            probs = softmax(scores)
            merged = model._merge_heads(np.matmul(probs, v))
            x = x + merged @ layer["wo"]
            x2, _ = layer_norm(x, layer["ln2_g"], layer["ln2_b"])
            h = np.maximum(x2 @ layer["fc1"] + layer["fc1_b"], 0.0)
            x = x + h @ layer["fc2"] + layer["fc2_b"]
        xf, _ = layer_norm(x, model.lnf_g, model.lnf_b)
        logits = xf[0, -1] @ model.lm_head + model.lm_bias
        logits = logits / max(temperature, 1e-6)
        p = softmax(logits)
        nxt = int(rng.choice(model.vocab, p=p))
        idx.append(nxt)
    return "".join(itos[i] for i in idx)


def main():
    p = argparse.ArgumentParser(description="nano-llm: train or sample a tiny transformer")
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--steps", type=int, default=400)
    tr.add_argument("--lr", type=float, default=3e-3)
    sm = sub.add_parser("sample")
    sm.add_argument("--prompt", default="An LLM ")
    sm.add_argument("--n", type=int, default=200)
    sm.add_argument("--temperature", type=float, default=0.8)
    sm.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if args.cmd == "train":
        model, stoi, itos, _ = train(steps=args.steps, lr=args.lr)
        print("--- sample ---")
        print(sample(model, stoi, itos, "An LLM ", n=220, temperature=0.7, seed=2))
    else:
        model, stoi, itos = NanoLM.load(DEFAULT_CKPT)
        print(sample(model, stoi, itos, args.prompt, args.n, args.temperature, args.seed))


if __name__ == "__main__":
    main()
