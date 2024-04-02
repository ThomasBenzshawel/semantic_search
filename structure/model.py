from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, vocab_size, embedding_size, dropout_p=0.1):
        super(EncoderTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)
        self.transformer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=8)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output = self.transformer(embedded)
        return output
    
class DecoderTransformer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderTransformer, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.transformer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8)
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
        output = self.transformer(output, hidden)
        output = self.out(output)
        return output, hidden


