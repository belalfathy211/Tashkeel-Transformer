import torch
from tqdm import tqdm
from transformer.tashkeel_model import TashkeelModel
from tashkeel_dataset import TashkeelDataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

max_iters = 5000
eval_iters = 200
eval_interval = 500
learning_rate = 3e-4

block_size = 256
n_embd = 384
n_head = 6
n_layer = 6

batch_size = 64
path = '/home/belal/projects/Tashkeel/Sadeed_Tashkeela/data'
data = TashkeelDataset(path, batch_size, block_size)

tashkeel_vocab_size = len(data.tashkeel_set)
chars_vocab_size = len(data.chars_set)

@torch.no_grad()
def estimate_loss(eval_iters = eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            kalam, X, Y = data.get_batch(split)
            logits, loss = model(kalam, X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = TashkeelModel(n_embd, n_head, block_size, n_layer, tashkeel_vocab_size, chars_vocab_size).to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

mn_val_loss = float('inf')
for iter in tqdm(range(max_iters)):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(eval_iters)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < mn_val_loss:
            mn = losses['val']
            torch.save(model.state_dict(), 'tashkeel_transformer.pth')
            print(f"best model state_dict saved to tashkeel_transformer.pth in step {iter} as val loss is : {losses['val']:.4f}")

    kalam, tashkeel_x, tashkeel_y = data.get_batch('train')
    logits, loss = model(kalam, tashkeel_x, tashkeel_y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


