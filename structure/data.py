from __future__ import unicode_literals, print_function, division
from io import open
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
import utils
from utils import CONTEXT_LENGTH
import torch

import torch
from torch.utils.data import Dataset

class ContextDataset(Dataset):
    def __init__(self, folder, context_length=512):
        documents = []
        for filename in os.listdir(folder):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                documents.append(utils.unicodeToAscii(file.read().strip()))


        self.data = []
        for doc in documents:
            words = doc.split()
            for i in range(len(words) - context_length):
                context = ' '.join(words[i:i+context_length])
                target = ' '.join(words[i:i+context_length+1])
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_tensor = torch.tensor(self.tokenize_and_pad(context))
        target_tensor = torch.tensor(self.tokenize_and_pad(target))
        return context_tensor, target_tensor

    def tokenize_and_pad(self, text, max_length=512):
        # Tokenize the text and convert to token IDs
        tokenizer = utils.BytePairTokenizer()
        token_ids = tokenizer.tokenize(text)

        # Pad or truncate the token IDs to the desired length
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids += [0] * (max_length - len(token_ids))

        return token_ids