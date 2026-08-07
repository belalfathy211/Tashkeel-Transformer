import torch
from transformer.tashkeel_model import TashkeelModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'

block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
path = '/home/belal/projects/Tashkeel/Sadeed_Tashkeela/data'
batch_size = 64

checkpoint = torch.load('tashkeel_transformer.pth', map_location=device)
tashkeel_vocab_size = checkpoint['tashkeel_vocab_size']
chars_vocab_size = checkpoint['chars_vocab_size']
stoi_chars = checkpoint['stoi_chars']
stoi_tashkeel = checkpoint['stoi_tashkeel']
itos_tashkeel = checkpoint['itos_tashkeel']

def merg_tashkeel(text, tashkeel):
    out = ""
    for i, ch in enumerate(text):
        out+= ch
        if i < len(tashkeel) and tashkeel[i] != "":
            out += tashkeel[i]
    return out

def predict(kalam, model):
    start_seq = [stoi_tashkeel["<START>"]]
    input = [stoi_chars[ch] for ch in kalam]
    de_output = model.predict(input, start_seq)
    decoded_tashkeel_output = [itos_tashkeel[t] for t in de_output]
    return merg_tashkeel(kalam, decoded_tashkeel_output)

input_kalam = input("Enter your kalam without tashkeel: ")
fixed_input_kalam = ""
tashkeelat = {'َ', 'ُ', 'ِ', 'ً', 'ٌ', 'ٍ', 'ْ', 'ّ', 'َّ', 'ُّ', 'ِّ', 'ًّ', 'ٌّ', 'ٍّ'}
fixed_input_kalam = "".join([ch for ch in input_kalam if ch not in tashkeelat])
fixed_input_kalam = fixed_input_kalam[:256]

model = TashkeelModel(n_embd, n_head, block_size, n_layer, tashkeel_vocab_size, chars_vocab_size, dropout).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(predict(fixed_input_kalam , model))