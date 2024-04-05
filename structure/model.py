from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTEXT_LENGTH = 512

class EncoderTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads, context_length, dropout_p=0.1):
        super(EncoderTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.context_length = context_length
        CONTEXT_LENGTH = context_length
        self.dropout = nn.Dropout(dropout_p)
        self.pos_encoder = PositionalEncoding(embedding_size, dropout_p, context_length)
        transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=3)

    def forward(self, input):
        # The input is a tensor of shape (batch_size, seq_length)
        # Check the shape of the input tensor
        # print(input.shape, "input.shape")
        embedded = self.dropout(self.embedding(input))
#         print(embedded.shape)
        embedded = self.pos_encoder(embedded)
        #print the tensor
        # print(embedded)
        output = self.transformer(embedded)
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        pe = self.pe[:, :seq_len, :]
        pe = pe.expand(batch_size, -1, -1)
        
        x = x + pe
        return self.dropout(x)
    
class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, context_length=1024):
        super(DecoderTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.context_length = context_length
        self.pos_encoder = PositionalEncoding(hidden_size, 0.1, context_length)
        self.dropout = nn.Dropout(0.1)
        transformer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(transformer, num_layers=3)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.out.weight = nn.Parameter(self.embedding.weight)


    def forward(self, encoded_memory, target_tensor=None):
        target = self.embedding(target_tensor)
        target = self.pos_encoder(target)
        decoded_output = self.transformer_decoder(target, encoded_memory)
#         print(decoded_output.shape)
        decoder_output = self.layernorm(decoded_output)
        decoder_output = self.out(decoded_output)
        return decoder_output


