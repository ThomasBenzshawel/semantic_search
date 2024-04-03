import torch

# Load the saved model
model = torch.load('path/to/saved/model.pth')

# Set the model to evaluation mode
model.eval()

# Disable gradient computation
with torch.no_grad():
    # Prepare your input data
    input_data = "The quick brown fox jumps over the lazy dog"

    output = model(input_data)