import torch
import utils


def predict(input_data):
    
    # Load the saved model
    model = torch.load('model.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation
    with torch.no_grad():
        tokenizer = utils.BytePairTokenizer()
        input_data = tokenizer.tokenize(utils.unicodeToAscii(input_data))
        output = model(input_data)
    
    return output


def vectorize_document(document):
    # Vectorize the document using the same tokenizer used during training
    tokenizer = utils.BytePairTokenizer()
    #since we can't read in the whole file at once, we need to read in the file in chunks, use utils.CONTENT_LENGTH to read in the file in chunks

    # Load the saved model
    model = torch.load('model.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation
    with torch.no_grad():
        input_data = tokenizer.tokenize(utils.unicodeToAscii(input_data))
        output = model(input_data)

        tensor_list = []
        for i in range(0, len(document), tokenizer.C):
            input = tokenizer.tokenize(utils.unicodeToAscii(document[i:i+utils.CONTEXT_LENGTH]))
            input_tensor = torch.tensor(input, dtype=torch.long)
            tensor_list.append(input_tensor)
    
        #take the item by item average of the tensor list

        input_tensor = torch.stack(tensor_list).mean(dim=0)

        return input_tensor