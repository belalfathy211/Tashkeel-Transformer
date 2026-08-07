import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer.modules import MultiHeadAttention, FeedFoward
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size_decoder, dropout, masked=True):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size_decoder, dropout, masked=masked)
        self.ca = MultiHeadAttention(n_embd, n_head, head_size, block_size_decoder, dropout, masked=False)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x, encoder_kv):
        x = x + self.sa(self.ln1(x))
        x = x + self.ca(self.ln2(x), encoder_kv=encoder_kv)
        x = x + self.ffwd(self.ln3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, n_embd, n_head, block_size_decoder, n_layer, dropout, tashkeel_vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(tashkeel_vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size_decoder, n_embd)
        self.blocks = nn.Sequential(*[DecoderBlock(n_embd, n_head, block_size_decoder, dropout, masked=True) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, tashkeel_vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, encoder_kv= None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x, encoder_kv=encoder_kv)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
            accuracy = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

            corrects = sum(torch.argmax(logits,1) == targets).item()
            accuracy = corrects / targets.numel()

        return logits, loss, accuracy
