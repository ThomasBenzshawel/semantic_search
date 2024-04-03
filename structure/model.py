from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO improve the tokenization process by using byte pair encoding instead of one hot encoding

MAX_OUTPUT_LENGTH = 10

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embeding_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = embeding_size

        self.embedding = nn.Embedding(vocab_size, embeding_size)
        self.gru = nn.GRU(embeding_size, embeding_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_OUTPUT_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
    

#Insead of using an RNN, we can use a transformer architecture to encode the input and improve the model's performance
class EncoderTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads, context_length, dropout_p=0.1):
        super(EncoderTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.context_length = context_length
        self.dropout = nn.Dropout(dropout_p)
        self.pos_encoder = PositionalEncoding(embedding_size, dropout_p, context_length)
        transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=6)

    def forward(self, input):
        # The input is a tensor of shape (batch_size, seq_length)
        # Check the shape of the input tensor
        # print(input.shape, "input.shape")
        embedded = self.dropout(self.embedding(input))
        embedded = self.pos_encoder(embedded)
        #print the tensor
        # print(embedded)
        output = self.transformer(embedded)
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, context_length=1024):
        super(DecoderTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.context_length = context_length
        self.pos_encoder = PositionalEncoding(hidden_size, 0.1, context_length)
        self.dropout = nn.Dropout(0.1)
        transformer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(transformer, num_layers=6)
        self.out = nn.Linear(hidden_size, vocab_size)


    def forward(self, encoded_memory, target_tensor=None):
        target = self.embedding(target_tensor)
        target = self.pos_encoder(target)
        decooded_output = self.transformer_decoder(target, encoded_memory)
        decoder_output = self.out(decooded_output)
        return decoder_output


