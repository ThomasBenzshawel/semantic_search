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

                #TODO Change the target to be the next word in the sequence or a paraphrase of the input
                # target = filename.split("/")[-1].split(".")[0]
                # tokenized_target = tokenizer.tokenize(utils.unicodeToAscii(target))
                # self.name_list.append(tokenized_target)

                #TODO Change the target to be the next word in the sequence



                

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        #data is a pair of input and target
        #We want to predict the target from the input
        #The input must be split into chunks of CONTEXT_LENGTH of tokens

        #Target is the next word in the sequence 
        input = self.input_list[idx]
        # Inpout must be split into chunks of CONTEXT_LENGTH, so we can predict the next word in the sequence
        input_tensor_document = torch.tensor(input, dtype=torch.long)

        # Grab the target tensor from the input tensor by shifting the input tensor by one
        target_tensor = input_tensor_document.clone()
        target_tensor = target_tensor[1:]

        #input is currently a whole document, we need to split it into chunks of CONTEXT_LENGTH

        # Split the input tensor into chunks of CONTEXT_LENGTH
        input_tensor = torch.split(input_tensor_document, CONTEXT_LENGTH)
        #turn the chunks into a list
        input_tensor = list(input_tensor)
        #Check the shape
        # print(input_tensor[0].shape, "input_tensor[0].shape")
        #fill the last chunk with zeros if it is less than CONTEXT_LENGTH
        if len(input_tensor[-1]) < CONTEXT_LENGTH:
            input_tensor[-1] = F.pad(input_tensor[-1], (0, CONTEXT_LENGTH - len(input_tensor[-1])), "constant", 0)

        # Split the target tensor into chunks of CONTEXT_LENGTH
        target_tensor = torch.split(target_tensor, CONTEXT_LENGTH)

        #turn the chunks into a list
        target_tensor = list(target_tensor)

        #fill the last chunk with zeros if it is less than CONTEXT_LENGTH
        if len(target_tensor[-1]) < CONTEXT_LENGTH:
            target_tensor[-1] = F.pad(target_tensor[-1], (0, CONTEXT_LENGTH - len(target_tensor[-1])), "constant", 0)


        #Check the shape
        # print(target_tensor[0].shape, "target_tensor[0].shape")

        #Stack the input and target tensors
        # print(input_tensor[0].shape, "input_tensor[0].shape")
        # print(len(input_tensor), "len(input_tensor)")

        #input tensor is a list of tensors, each tensor is of shape (1, CONTEXT_LENGTH)
        # We stack the tensors to create a 2D tensor of shape (batch_size, CONTEXT_LENGTH) 
        input_tensor = torch.stack(input_tensor, dim=0)
        target_tensor = torch.stack(target_tensor, dim=0)

        # print(input_tensor.shape, "input_tensor.shape")

        return input_tensor, target_tensor