import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 64
block_size = 256
max_iters = 5000
eval_iters = 200
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)


with open('/home/belal/projects/Tashkeel/Sadeed_Tashkeela/data/input.txt', 'r', encoding='utf-8') as f:
    input = f.read()

with open('/home/belal/projects/Tashkeel/Sadeed_Tashkeela/data/output.txt', 'r', encoding='utf-8') as f:
    output = f.read()

#_________________________________________

chars = sorted(list(set(input)))
if '<UNK>' not in chars:
    chars.append('<UNK>')

chars_stoi = { ch:i for i,ch in enumerate(chars) }
chars_itos = { i:ch for i,ch in enumerate(chars) }
chars_vocab_size = len(chars)
chars_unk_id = chars_stoi['<UNK>']
encode_chars = lambda s: [chars_stoi.get(c, chars_unk_id) for c in s]

#_________________________________________
tashkeel_without_shaddah = {"\u064E", "\u064F", "\u0650", "\u0652", "\u064B", "\u064C", "\u064D", "\u0651"}
tashkeel_set = ["<UNK>", "<START>", "<END>", "", "\u064E", "\u064F", "\u0650", "\u0652", "\u064B", "\u064C", "\u064D", "\u0651", "\u0651\u064E", "\u0651\u064F", "\u0651\u0650", "\u0651\u064B", "\u0651\u064C", "\u0651\u064D", "\u0651\u0652"]

def extract_tashkeel(text):
    out = []
    curr = ""
    for ch in text:
        if ch in tashkeel_without_shaddah:
            curr+= ch
        else:
            out.append(curr)
            curr = ""
    out.append(curr)
    return out[1:]

def merg_tashkeel(text, tashkeel):
    out = ""
    i = 0
    for i, ch in enumerate(text):
        out+= ch
        if i < len(tashkeel) and tashkeel[i] != "":
            out += tashkeel[i]
    return out

tashkeel_stoi = {t: i for i, t in enumerate(tashkeel_set)}
tashkeel_itos = {i: t for i, t in enumerate(tashkeel_set)}
tashkeel_vocab_size = len(tashkeel_set)
tashkeel_unk_id = tashkeel_stoi['<UNK>']
encode_tashkeel = lambda s: [tashkeel_stoi.get(c, tashkeel_unk_id) for c in s]
decode_tashkeel = lambda l: [tashkeel_itos.get(i) for i in l]

#_________________________________________

tashkeel_data = torch.tensor(encode_tashkeel(extract_tashkeel(output)), dtype=torch.long)
input_data = torch.tensor(encode_chars(input), dtype=torch.long)

n = int(0.9*len(tashkeel_data))
train_tashkeel_data = tashkeel_data[:n]
val_tashkeel_data = tashkeel_data[n:]
train_input_data = input_data[:n]
val_input_data = input_data[n:]

def get_batch(split):
    tashkeel_data = train_tashkeel_data if split == 'train' else val_tashkeel_data
    input_data = train_input_data if split == 'train' else val_input_data
    ix = torch.randint(min(len(tashkeel_data),len(input_data)) - block_size, (batch_size,))
    kalam_x_encode = torch.stack([input_data[i:i+block_size] for i in ix])
    tashkeel_x_decode = torch.stack([tashkeel_data[i:i+block_size] for i in ix])
    tashkeel_y_decode = torch.stack([tashkeel_data[i+1:i+block_size+1] for i in ix])
    kalam, tashkeel_x, tashkeel_y = kalam_x_encode.to(device), tashkeel_x_decode.to(device), tashkeel_y_decode.to(device)
    return kalam, tashkeel_x, tashkeel_y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            kalam, X, Y = get_batch(split)
            logits, loss = model(kalam, X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size, masked = True):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
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
    def __init__(self, num_heads, head_size, masked=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, masked=masked) for _ in range(num_heads)])
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

class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, masked=False):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, masked=masked)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, masked=True):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, masked=masked)
        self.ca = MultiHeadAttention(n_head, head_size, masked=False)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x, encoder_kv):
        x = x + self.sa(self.ln1(x))
        x = x + self.ca(self.ln2(x), encoder_kv=encoder_kv)
        x = x + self.ffwd(self.ln3(x))
        return x

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(chars_vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[EncoderBlock(n_embd, n_head=n_head, masked=False) for _ in range(n_layer)])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(tashkeel_vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[DecoderBlock(n_embd, n_head=n_head, masked=True) for _ in range(n_layer)])
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
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

class TashkeelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, kalam_x, tashkeel_x, target = None):
        encoder_out = self.encoder(kalam_x)
        logits, loss = self.decoder(tashkeel_x,  targets=target, encoder_kv=encoder_out)
        return logits, loss


model = TashkeelModel().to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    kalam, tashkeel_x, tashkeel_y = get_batch('train')
    logits, loss = model(kalam, tashkeel_x, tashkeel_y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save({
        'model_state_dict': model.state_dict(),
        'chars_stoi': chars_stoi,
        'tashkeel_stoi': tashkeel_stoi,
    }, 'tashkeel_transformer1.pth')

print("Model saved to tashkeel_transformer1.pth")