import torch
from transformer.tashkeel_model import TashkeelModel
from tashkeel_dataset import TashkeelDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

block_size = 256
n_embd = 384
n_head = 6
n_layer = 6

path = '/home/belal/projects/Tashkeel/Sadeed_Tashkeela/data'
batch_size = 64
data = TashkeelDataset(path, batch_size, block_size)
tashkeel_vocab_size = len(data.tashkeel_set)
chars_vocab_size = len(data.chars_set)

def merg_tashkeel(text, tashkeel):
    out = ""
    for i, ch in enumerate(text):
        out+= ch
        if i < len(tashkeel) and tashkeel[i] != "":
            out += tashkeel[i]
    return out

def predict(kalam, model):
    start_seq = data.encode_tashkeel(["<START>"])
    input = data.encode_kalam(kalam)
    de_output = model.predict(input, start_seq)
    return merg_tashkeel(kalam, data.decode_tashkeel(de_output))

model = TashkeelModel(n_embd, n_head, block_size, n_layer, tashkeel_vocab_size, chars_vocab_size).to(device)
model.load_state_dict(torch.load('tashkeel_transformer.pth'))
model.eval()
print(predict("أصحهما في الشرح الصغير الأول وكلام الأصل يميل إليه" , model))

