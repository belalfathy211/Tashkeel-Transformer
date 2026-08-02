import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer.encoder import Encoder
from transformer.decoder import Decoder
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

class TashkeelModel(nn.Module):
    def __init__(self, n_embd, n_head, block_size, n_layer, tashkeel_vocab_size, chars_vocab_size):
        super().__init__()
        self.encoder = Encoder(n_embd, n_head, block_size, n_layer, chars_vocab_size)
        self.decoder = Decoder(n_embd, n_head, block_size+1, n_layer, tashkeel_vocab_size)

    def forward(self, kalam_x, tashkeel_x, target = None):
        encoder_out = self.encoder(kalam_x)
        logits, loss = self.decoder(tashkeel_x,  targets=target, encoder_kv=encoder_out)
        return logits, loss

    def predict(self, encoded_input, start_seq):
        de_output = start_seq
        x = self.encoder(torch.tensor(encoded_input).unsqueeze(0).to(device))
        for i in range(len(encoded_input)):
            logits, _ = self.decoder(torch.tensor(de_output).unsqueeze(0).to(device), encoder_kv=x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            next_tashkeel = torch.argmax(probs).item()
            de_output.append(int(next_tashkeel))
        return de_output[1:]