import torch
from torchsummary import summary
"""not working ywt"""
# Load the saved model architecture and weights
model = torch.load('Models/model_pre_training_1')
model.eval()

# Print the model architecture
print("Model Architecture:\n")
#print(model)
summary(model, input_size=(1,128))

# Print the number of parameters and their types
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

#print("\nNumber of Trainable Parameters:", num_trainable_params)
#print("Number of Non-Trainable Parameters:", num_non_trainable_params)

# Print the device where the model is located
device = next(model.parameters()).device
#print("\nDevice:", device)