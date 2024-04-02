from __future__ import unicode_literals, print_function, division
from io import open
import os
import torch.nn.functional as F
from torch.utils.data import Dataset


class NotesDataset(Dataset):
    def __init__(self, folder, loadTest=False):
    # Iterate through the data in the data folder with the folder name specified
        self.folder = folder
        self.loadTest = loadTest
        self.data = []

        directory = os.path.join('data/', folder)
        for filename in os.listdir(directory):
            f = os.path.join(directory + "/", filename)
            # checking if it is a file
            if os.path.isfile(f):
                self.data.append(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        with open(self.data[idx], 'r') as file:
            input = file.read()
        
            target = self.data.split("/")[-1].split(".")[0]
            return input, target