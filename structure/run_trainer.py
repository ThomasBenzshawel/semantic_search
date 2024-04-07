from trainer import *
import torch
import os
from data import NotesDataset_Given_list
from model import EncoderTransformer, DecoderTransformer
import utils



def train():
    print("=======================================================")
    print("Pretrain Encoder with decoder for structure model")
    print("=======================================================")

    #make a folder for the model
    folder = "model_structure"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Load the dataset 
    # Path to the dataset
    path = "/home/benzshawelt/NLP/semantic_search/data/data_zipped/literature"
    dataset = NotesDataset(path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16)

    # Define the model
    embedding_size = 128
    num_heads = 8
    tokenizer = utils.BytePairTokenizer()
    vocab_size = tokenizer.n_words
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EncoderTransformer(vocab_size, embedding_size, num_heads, context_length=512).to(device)
    decoder = DecoderTransformer(vocab_size, embedding_size, num_heads, context_length=512).to(device)

    # Train the model
    n_epochs = 100
    print_every = 10
    plot_every = 10

    train(dataloader, encoder, decoder, 100)

    print("=======================================================")
    print("Training complete")
    print("=======================================================")



def train_preloaded_data(documents, num_epochs=10, print_every=1):
    #documents is a list of strings
    folder = "temp_structure"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Load the dataset 
    # Path to the dataset
    dataset = NotesDataset_Given_list(documents)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)

    # Define the model
    embedding_size = 128
    num_heads = 8
    tokenizer = utils.BytePairTokenizer()
    vocab_size = tokenizer.n_words
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EncoderTransformer(vocab_size, embedding_size, num_heads, context_length=512).to(device)
    decoder = DecoderTransformer(vocab_size, embedding_size, num_heads, context_length=512).to(device)


    model = train_return_model(dataloader, encoder, decoder, n_epochs=num_epochs, save_folder=folder)

    return model






if __name__ == "__main__":
    train()