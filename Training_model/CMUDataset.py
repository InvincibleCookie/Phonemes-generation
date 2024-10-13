import torch
from torch.utils.data import Dataset


class CMUDataset(Dataset):
    def __init__(self, words, phonemes, tokenizer, max_length=128):
        self.words = words
        self.phonemes = phonemes
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        phoneme = " ".join(self.phonemes[idx])

        input_ids = self.tokenizer(word, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True).input_ids
        labels = self.tokenizer(phoneme, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True).input_ids

        return {'input_ids': input_ids.squeeze(), 'labels': labels.squeeze()}

