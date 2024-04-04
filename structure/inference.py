import torch
import torch.nn.functional as F
import utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(input_data):
    
    # Load the saved model
    model = torch.load('saved_models/encoder_epoch_2.pth', map_location=device)

    # Move the model to the device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation
    with torch.no_grad():
        tokenizer = utils.BytePairTokenizer()
        input_data = tokenizer.tokenize(utils.unicodeToAscii(input_data))
        input_data = torch.tensor(input_data, dtype=torch.long)
        input_data = input_data.to(device)
        output = model(input_data)

    #model output needs to be a 1 x embedding size tensor
    output = output.mean(dim=0)
    return output


def vectorize_document(document):
    # Vectorize the document using the same tokenizer used during training
    tokenizer = utils.BytePairTokenizer()
    #since we can't read in the whole file at once, we need to read in the file in chunks, use utils.CONTENT_LENGTH to read in the file in chunks

    # Load the saved model
    model = torch.load('saved_models/encoder_epoch_2.pth', map_location=device)

    # Move the model to the device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation
    with torch.no_grad():
        input_data = tokenizer.tokenize(utils.unicodeToAscii(document))
    
        tensor_list = []
        for i in range(0, len(input_data), utils.CONTEXT_LENGTH):
            input = input_data[i:i+utils.CONTEXT_LENGTH]
            input_tensor = torch.tensor(input, dtype=torch.long)
            input_tensor = input_tensor.to(device)
            model_output = model(input_tensor)

            #model output needs to be a 1 x embedding size tensor
            #Average the output of the model
            model_output = model_output.mean(dim=0)

            tensor_list.append(model_output)
    
        #take the item by item average of the tensor list
        model_output = torch.stack(tensor_list).mean(dim=0)

        #model output needs to be a 1 x embedding size tensor
        return model_output
    
if __name__ == "__main__":
    input_data = "This is a test document."
    output = predict(input_data)
    print(output)
    document = "This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. and so on and more text. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. and so on and more text. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. and so on and more text. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. and so on and more text. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. and so on and more text. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. and so on and more text. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. and so on and more text. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. and so on and more text. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. and so on and more text. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. This is a test document. This is a test document. This is a test document. We need to make this super long to test the model. So we will keep adding more text to this document. and so on and more text."
    input_tensor = vectorize_document(document)
    print(input_tensor)