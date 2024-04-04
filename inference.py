import torch
import utils


def predict(input_data):
    
    # Load the saved model
    model = torch.load('encoder_epoch_2.pth')

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
    # model = torch.load('encoder_epoch_2.pth')
    model = torch.load('encoder_epoch_2.pth', map_location=torch.device('cpu')) #my GPU isnt set up for CUDA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation
    with torch.no_grad():
        input_data = torch.tensor(tokenizer.tokenize(utils.unicodeToAscii(document)))

        tensor_list = []
        for i in range(0, len(document), utils.CONTEXT_LENGTH):
            input = tokenizer.tokenize(utils.unicodeToAscii(document[i:i+utils.CONTEXT_LENGTH]))
            input_tensor = torch.tensor(input, dtype=torch.long)
            input_tensor = input_tensor.to(device)
            model_output = model(input_tensor)
            tensor_list.append(model_output)
    
        #take the item by item average of the tensor list
        print([len(t) for t in tensor_list])

        input_tensor = torch.stack(tensor_list).mean(dim=0)

        return input_tensor