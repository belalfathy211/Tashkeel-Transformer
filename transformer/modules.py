import torch
import torch.nn as nn
from torch.nn import functional as F

n_embd = 384
dropout = 0.2

class Head(nn.Module):
    def __init__(self, head_size, block_size_, masked = True):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size_, block_size_)))
        self.dropout = nn.Dropout(dropout)
        self.masked = masked

    def forward(self, x, encoder_kv= None):
        B,T,C = x.shape
        q = self.query(x)

        if encoder_kv is None:
            k = self.key(x)
            v = self.value(x)
        else:
            k = self.key(encoder_kv)
            v = self.value(encoder_kv)

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5

        if self.masked:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, block_size_, masked=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, block_size_, masked=masked) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_kv = None):
        out = torch.cat([h(x, encoder_kv=encoder_kv) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

