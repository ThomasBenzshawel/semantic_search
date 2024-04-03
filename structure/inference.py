import torch


def predict(input_data):
    
    # Load the saved model
    model = torch.load('path/to/saved/model.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation
    with torch.no_grad():
        output = model(input_data)
    
    return output