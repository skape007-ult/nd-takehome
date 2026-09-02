"""Decoder-only transformer, from scratch. Reference size: 4 layers, d=256,
8 heads, ~3.3M params. No pretrained weights anywhere.

Standard pre-norm GPT: learned token + position embeddings, causal multi-head
attention, GELU MLP, weight-tied LM head. Kept deliberately small and plain --
bigger is not better for this task, and simple is easy to defend.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 101
    n_layer: int = 4
    n_head: int = 8
    d_model: int = 256
    d_ff: int = 1024
    block_size: int = 256
    dropout: float = 0.1
    pad_id: int = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop.p if self.training else 0.0,
            is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f'sequence {T} > block {self.cfg.block_size}'
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, stop_id, eos_id,
                 temperature=0.0, greedy=True, generator=None):
        """Autoregressive decode. Stops when every row has emitted `stop_id`
        (QED) or `eos_id`, or after max_new_tokens. idx is (B, T0). Returns the
        full id tensor including the prompt."""
        self.eval()
        B = idx.size(0)
        done = torch.zeros(B, dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(cond)
            logits = logits[:, -1, :]
            if greedy or temperature <= 0:
                nxt = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.cfg.pad_id), nxt)
            idx = torch.cat([idx, nxt[:, None]], dim=1)
            done = done | (nxt == stop_id) | (nxt == eos_id)
            if bool(done.all()):
                break
        return idx


def build_model(vocab_size, **overrides):
    cfg = GPTConfig(vocab_size=vocab_size, **overrides)
    return GPT(cfg)


def load_ckpt(path, device='cpu'):
    """Reconstruct a model from a training checkpoint and put it in eval mode."""
    blob = torch.load(path, map_location=device)
    cfg = GPTConfig(**blob['cfg'])
    model = GPT(cfg).to(device)
    model.load_state_dict(blob['model_state'])
    model.eval()
    return model, blob
