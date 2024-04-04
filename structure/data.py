from __future__ import unicode_literals, print_function, division
from io import open
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
import utils
from utils import CONTEXT_LENGTH
import torch


# class NotesDataset(Dataset):
#     def __init__(self, directory, loadTest=False):
#     # Iterate through the data in the data folder with the folder name specified
#         self.loadTest = loadTest
#         self.data_list = []
#         # self.name_list = []
#         self.input_list = []

        
#         for filename in os.listdir(directory):
#             f = os.path.join(directory + "/", filename)
#             # checking if it is a file
#             if os.path.isfile(f):
#                 self.data_list.append(f)

#             with open(f, 'r', encoding="utf_8") as file:
#                 #Read in as utf-8
#                 tokenizer = utils.BytePairTokenizer()
#                 input = file.read().strip()
#                 tokenized_input = tokenizer.tokenize(utils.unicodeToAscii(input))
#                 self.input_list.append(tokenized_input)


#     def __len__(self):
#         return len(self.data_list)
    
#     def __getitem__(self, idx):
#         #data is a pair of input and target
#         #We want to predict the target from the input
#         #The input must be split into chunks of CONTEXT_LENGTH of tokens

#         #Target is the next word in the sequence 
#         input = self.input_list[idx]
#         # Input must be split into overlapping chunks of CONTEXT_LENGTH, so we can predict the next word in the sequence
        
#         input_tensor_list = []
#         target_tensor_list = []

#         #if the input is less than the context length, we can't  iterate over it, so we grab it and then pad it with zeros to the context length
#         if len(input) < CONTEXT_LENGTH:
#             input_tensor = torch.tensor(input, dtype=torch.long)
#             #Shift the target tensor by one
#             target_tensor = torch.tensor(input[1:], dtype=torch.long)
#             #pad the input and target tensors
#             input_tensor = F.pad(input_tensor, (0, CONTEXT_LENGTH - len(input)), value=0)
#             #since the target tensor is shifted by one, we need to pad it with one less than the context length
#             target_tensor = F.pad(target_tensor, (0, CONTEXT_LENGTH - len(input) + 1), value=0)

#             input_tensor_list.append(input_tensor)
#             target_tensor_list.append(target_tensor)

#         else:
            
#             #iterate over for overlapping chunks of CONTEXT_LENGTH
#             for i in range(len(input) - CONTEXT_LENGTH):
#                 #Get the input tensor
#                 input_tensor = torch.tensor(input[i:i+CONTEXT_LENGTH], dtype=torch.long)
#                 #Get the target tensor
#                 target_tensor = torch.tensor(input[i+1:i+CONTEXT_LENGTH + 1], dtype=torch.long)

#                 #if the tensor is size 513, we need to remove the last element
#                 if input_tensor.shape[0] == CONTEXT_LENGTH + 1:
#                     input_tensor = input_tensor[:-1]
#                     target_tensor = target_tensor[:-1]

#                 input_tensor = F.pad(input_tensor, (0, CONTEXT_LENGTH), value=0)
#                 target_tensor = F.pad(target_tensor, (0, CONTEXT_LENGTH), value=0)

                
#                 input_tensor_list.append(input_tensor)
#                 target_tensor_list.append(target_tensor)



#         #Stack the tensors ontop of each other
#         input_tensor = torch.stack(input_tensor_list)
#         target_tensor = torch.stack(target_tensor_list)

#         return input_tensor, target_tensor

class NotesDataset(Dataset):
    def __init__(self, directory, context_length=512):
        self.data = []
        documents = []
        for filename in os.listdir(directory):
            f = os.path.join(directory + "/", filename)
            # checking if it is a file
            if os.path.isfile(f):
            
                with open(f, 'r', encoding="utf_8") as file:
                    #Read in as utf-8
                    input = file.read().strip()
                    documents.append(utils.unicodeToAscii(input))
        

        for doc in documents:
            words = doc.split()
            for i in range(len(words) - context_length):
                context = ' '.join(words[i:i+context_length])
                target = ' '.join(words[i:i+context_length+1])
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_tensor = torch.tensor(self.tokenize_and_pad(context))
        target_tensor = torch.tensor(self.tokenize_and_pad(target))
        return context_tensor, target_tensor

    def tokenize_and_pad(self, text, max_length=512):
        # Tokenize the text and convert to token IDs
        tokenizer = utils.BytePairTokenizer()
        token_ids = tokenizer.tokenize(text)

        # Pad or truncate the token IDs to the desired length
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids += [0] * (max_length - len(token_ids))

        return token_ids