import pandas as pd
import os
from tqdm import tqdm
from io import StringIO
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TashkeelDataset:
    def __init__(self, path, batch_size = 64, block_size = 256):
        print("Start read tashkeeldataset....")
        all_input_data = StringIO()
        all_output_data = StringIO()
        self.batch_size = batch_size
        self.block_size = block_size
        for f in os.listdir(path):
            if not f.endswith('.parquet'):
                continue
            print(f"take the {f} ....")
            pq_path = path + '/' + f
            input_file = pd.read_parquet(pq_path)["input"]
            output_file = pd.read_parquet(pq_path)["output"]
            for s in tqdm(input_file):
                if type(s) == str:
                    all_input_data.write(s)
            for s in tqdm(output_file):
                if type(s) == str:
                    all_output_data.write(s)
        self.all_input_data = all_input_data.getvalue()
        self.all_output_data = all_output_data.getvalue()
        print("Initializing Tashkeel... ")
        self.initialize_tashkeel()
        print("Initializing Chars... ")
        self.initialize_chars(self.all_input_data)
        print("Initialize All Data... ")
        self.initialize_all_data()


    def _extract_tashkeel(self, text):
        tashkeel_without_shaddah = {"\u064E", "\u064F", "\u0650", "\u0652", "\u064B", "\u064C", "\u064D", "\u0651"}
        out = []
        curr = ""
        for ch in tqdm(text):
            if ch in tashkeel_without_shaddah:
                curr += ch
            else:
                out.append(curr)
                curr = ""

        out.append(curr)
        return out[1:]

    def initialize_tashkeel(self):
        self.tashkeel_set = ["<UNK>", "<START>", "<END>", "", "\u064E", "\u064F", "\u0650", "\u0652", "\u064B", "\u064C",
                        "\u064D", "\u0651",
                        "\u0651\u064E", "\u0651\u064F", "\u0651\u0650", "\u0651\u064B", "\u0651\u064C", "\u0651\u064D",
                        "\u0651\u0652"]
        self.stoi_tashkeel = {ch: i for i, ch in enumerate(self.tashkeel_set)}
        self.itos_tashkeel = {i: ch for i, ch in enumerate(self.tashkeel_set)}

    def encode_tashkeel(self, tashkeels):
        tashkeel_encoded = []
        for t in tqdm(tashkeels):
            if t in self.tashkeel_set:
                tashkeel_encoded.append(self.stoi_tashkeel[t])
            else:
                tashkeel_encoded.append(self.stoi_tashkeel['<UNK>'])

        return tashkeel_encoded

    def decode_tashkeel(self, ids):
        tashkeel_decoded = []
        for i in tqdm(ids):
            tashkeel_decoded.append(self.itos_tashkeel[i])

        return tashkeel_decoded

    def initialize_chars(self, all_data):
        self.chars_set = sorted(list(set(all_data)))
        self.chars_set.append('<UNK>')
        self.stoi_chars = {ch: i for i, ch in enumerate(self.chars_set)}

    def encode_kalam(self, kalam):
        kalam_encoded = []
        for ch in tqdm(kalam):
            if ch in self.chars_set:
                kalam_encoded.append(self.stoi_chars[ch])
            else:
                kalam_encoded.append(self.stoi_chars['<UNK>'])

        return kalam_encoded

    def initialize_all_data(self):
        tashkeels = self._extract_tashkeel(self.all_output_data)
        tashkeels_encoded = torch.tensor(self.encode_tashkeel(tashkeels), dtype=torch.long)
        input_encoded = torch.tensor(self.encode_kalam(self.all_input_data), dtype=torch.long)

        n = int(0.9 * len(tashkeels))
        self.train_tashkeel_data = tashkeels_encoded[:n]
        self.val_tashkeel_data = tashkeels_encoded[n:]
        self.train_input_data = input_encoded[:n]
        self.val_input_data = input_encoded[n:]

    def get_batch(self, split):
        tashkeel_data = self.train_tashkeel_data if split == 'train' else self.val_tashkeel_data
        input_data = self.train_input_data if split == 'train' else self.val_input_data
        ix = torch.randint(min(len(input_data), len(tashkeel_data)) - self.block_size, (self.batch_size,))
        input_without_tashkeel = torch.stack([input_data[i: i+self.block_size] for i in ix])
        x = torch.stack([torch.cat((torch.tensor([1]), tashkeel_data[i: i+self.block_size])) for i in ix])
        y = torch.stack([torch.cat((tashkeel_data[i: i+self.block_size], torch.tensor([2]))) for i in ix])
        return input_without_tashkeel.to(device), x.to(device), y.to(device)

if __name__ == '__main__':
    dataset_path = '/home/belal/projects/Tashkeel/Sadeed_Tashkeela/data'
    tashkeel_data = TashkeelDataset(dataset_path)
    kalam, x, y = tashkeel_data.get_batch('train')
    print(tashkeel_data.encode_tashkeel(['َ','ِ','ُ']))
    print(tashkeel_data.encode_kalam('اثجحخد ذ'))
    print(tashkeel_data.decode_tashkeel([3,6,7,1,3]))