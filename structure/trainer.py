from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
from torch import optim
import time
from model import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):
    
    tokenizer = BytePairTokenizer()
    vocab_size = tokenizer.n_words

    total_loss = 0
    for data in dataloader:
        
        input_tensor = data[0].to(device)
        #chagnge shape
        input_tensor = input_tensor.view(-1, CONTEXT_LENGTH)
        target_tensor = data[1].to(device)
        #change shape
        target_tensor = target_tensor.view(-1, CONTEXT_LENGTH)
        #check shape
        # print(input_tensor.shape, "input shape")
        # print(target_tensor.shape, "target shape")

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        #Using batching
        encoder_outputs = encoder(input_tensor)
        decoder_outputs = decoder(encoder_outputs, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, vocab_size),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))
            # Save the model to saved_models folder
            torch.save(encoder, 'saved_models/encoder_epoch_'+ str(epoch) +'.pth')
            torch.save(decoder, 'saved_models/decoder_epoch_'+ str(epoch) +'.pth')
            
def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        tokenizer = BytePairTokenizer()
        input_tensor = tokenizer.tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluateRandomly(encoder, decoder, input, output, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input, output)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

    