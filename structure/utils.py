from __future__ import unicode_literals, print_function, division
from io import open
import torch

from model import EOS_token
from model import SOS_token
import unicodedata
import tiktoken

# Use byte pair encoding instead of one hot encoding
class BytePairTokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.dec = tiktoken.get_decoding("cl100k_base")
        self.n_words = self.enc.vocab_size

    def tokenize(self, sentence):
        return self.enc.encode(sentence)
    
    def detokenize(self, tokens):
        return self.dec.decode(tokens)
    
    def tensorFromSentence(self, sentence, device):
        indexes = self.tokenize(sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)
    
    def tensorsFromPair(self, pair):
        input_tensor = self.tensorFromSentence(pair[0])
        target_tensor = self.tensorFromSentence(pair[1])
        return (input_tensor, target_tensor)
    
    def indexesFromSentence(self, sentence):
        return self.tokenize(sentence)
    

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )