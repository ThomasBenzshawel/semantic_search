from trainer import train
import torch
import os
from data import NotesDataset
from model import EncoderTransformer, DecoderTransformer
import utils

if __name__ == "__main__":

    print("=======================================================")
    print("Pretrain Encoder with decoder for structure model")
    print("=======================================================")

    #make a folder for the model
    folder = "model_structure"
    if not os.path.exists(folder):
        os.makedirs(folder)

    dataset = NotesDataset("literature", loadTest=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    # Define the model
    
    embedding_size = 256
    num_heads = 8
    tokenizer = utils.BytePairTokenizer()
    vocab_size = tokenizer.n_words
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EncoderTransformer(vocab_size, embedding_size, num_heads, context_length=1024).to(device)
    decoder = DecoderTransformer(vocab_size, embedding_size, num_heads).to(device)

    # Train the model
    n_epochs = 100
    print_every = 10
    plot_every = 10

    train(dataloader, encoder, decoder, 100)
    
    print("=======================================================")
    print("Training complete")
    print("=======================================================")

        