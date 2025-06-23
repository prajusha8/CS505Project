#SYSTEM IMPORTS
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys

_cd_ = os.path.dirname(os.path.abspath(__file__))
for _dir_ in [_cd_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_

#prepares the datasets for lstm training
class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        text = self.texts[i]
        label = self.labels[i]

        indices = []
        for word in text:
            index = self.vocab.get(word, 1)
            indices.append(index)

        indices = indices[:self.max_len]
        indices += [0] * (self.max_len - len(indices))

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long) #returns padded sequences and corresp. label as pt tensor