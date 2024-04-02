from trainer import train_epoch
import torch
import torch.nn as nn
import os
from data import NotesDataset
from model import EncoderRNN, DecoderRNN

if __name__ == "__main__":

    print("=======================================================")
    print("Pretrain structure generator with fixed viewpoints")
    print("=======================================================")

    #make a folder for the model
    folder = "model_structure"
    if not os.path.exists(folder):
        os.makedirs(folder)

    dataset = NotesDataset("data", loadTest=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    # Define the model
    input_size = 100
    hidden_size = 100
    output_size = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EncoderRNN(input_size, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, output_size).to(device)

    # Define the loss function
    criterion = nn.NLLLoss()

    # Train the model
    n_epochs = 100
    print_every = 10
    plot_every = 10

    train_epoch(dataloader, encoder, decoder, criterion)
    
    print("=======================================================")
    print("Training complete")
    print("=======================================================")

        