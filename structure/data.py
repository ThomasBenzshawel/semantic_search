from __future__ import unicode_literals, print_function, division
from io import open
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
import utils
from utils import CONTEXT_LENGTH
import torch


class NotesDataset(Dataset):
    def __init__(self, folder, loadTest=False):
    # Iterate through the data in the data folder with the folder name specified
        self.folder = folder
        self.loadTest = loadTest
        self.data_list = []
        # self.name_list = []
        self.input_list = []

        directory = os.path.join('data/', folder)
        for filename in os.listdir(directory):
            f = os.path.join(directory + "/", filename)
            # checking if it is a file
            if os.path.isfile(f):
                self.data_list.append(f)

            with open(f, 'r', encoding="utf_8") as file:
                #Read in as utf-8
                tokenizer = utils.BytePairTokenizer()
                input = file.read().strip()
                tokenized_input = tokenizer.tokenize(utils.unicodeToAscii(input))
                self.input_list.append(tokenized_input)


    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        #data is a pair of input and target
        #We want to predict the target from the input
        #The input must be split into chunks of CONTEXT_LENGTH of tokens

        #Target is the next word in the sequence 
        input = self.input_list[idx]
        # Input must be split into overlapping chunks of CONTEXT_LENGTH, so we can predict the next word in the sequence
        
        input_tensor_list = []
        target_tensor_list = []
        for i in range(0, len(input)-CONTEXT_LENGTH):
            #Get the input tensor
            input_tensor = torch.tensor(input[i:i+CONTEXT_LENGTH], dtype=torch.long)
            #Get the target tensor
            target_tensor = torch.tensor(input[i+1:i+CONTEXT_LENGTH+1], dtype=torch.long)

            input_tensor_list.append(input_tensor)
            target_tensor_list.append(target_tensor)

            

        #Stack the tensors ontop of each other
        input_tensor = torch.stack(input_tensor_list)
        target_tensor = torch.stack(target_tensor_list)

        return input_tensor, target_tensor