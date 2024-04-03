from __future__ import unicode_literals, print_function, division
import torch
import time
import math
import unicodedata
import tiktoken

SOS_token = 0
EOS_token = 1
CONTEXT_LENGTH = 1024

# Use byte pair encoding instead of one hot encoding
class BytePairTokenizer:
    def __init__(self, embedding_size=256):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.n_words = self.enc.n_vocab

    def tokenize(self, sentence):
        return self.enc.encode(sentence, allowed_special="all")
    
    def detokenize(self, tokens):
        return self.enc.decode(tokens)
    
    def tensorFromSentence(self, sentence):
        return torch.tensor(sentence, dtype=torch.long).view(1, -1)
    
    def tensorsFromPair(self, pair):
        input_tensor = self.tensorFromSentence(pair[0])
        target_tensor = self.tensorFromSentence(pair[1])
        return (input_tensor, target_tensor)
    
    def indexesFromSentence(self, sentence):
        return self.tokenize(sentence)
    
    def tokenizeToTensor(self, sentence):
        indexes = self.tokenize(sentence)
        return torch.tensor(indexes, dtype=torch.long).view(1, -1)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))