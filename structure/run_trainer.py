from trainer import train
import torch
import torch.nn as nn
import os
from data import NotesDataset
from model import EncoderRNN, DecoderRNN

if __name__ == "__main__":

    print("=======================================================")
    print("Pretrain Encoder with decoder for structure model")
    print("=======================================================")

    #make a folder for the model
    folder = "model_structure"
    if not os.path.exists(folder):
        os.makedirs(folder)

    dataset = NotesDataset("literature", loadTest=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    # Define the model
    input_size = 100
    hidden_size = 100
    output_size = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EncoderRNN(input_size, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, output_size).to(device)

    # Train the model
    n_epochs = 100
    print_every = 10
    plot_every = 10

    train(dataloader, encoder, decoder, 100)
    
    print("=======================================================")
    print("Training complete")
    print("=======================================================")

        