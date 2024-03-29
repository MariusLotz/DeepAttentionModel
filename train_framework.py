import torch
import time
import os
from Helper_Functions import data_table_to_tensors
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Train_model import train_model
from Model_classes.L2BinaryClassifier import L2BinaryClassifier

#models = {L2BinaryClassifier:[p1,p2,p3], L2BinaryClassifier2:[p1,p2,p3], }

def make_folder(folder_name, path='Models'):

    # Combine the parent directory path with the folder name to create the full path
    folder_path = os.path.join(path, folder_name)

    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # If it doesn't exist, create it
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created successfully at '{path}'")
    else:
        # If it already exists, print a message indicating that
        print(f"Folder '{folder_name}' already exists at '{path}'")


def train_all(models, dataset_dir='Example_Problems/Real_Data/UCRoutput'):
    for model_class, param_list in models.items():
        make_folder(model_class.__name__)

        # Iterate over each folder in the rdataset directory
        for folder_name in os.listdir(dataset_dir):
            folder_path = os.path.join(dataset_dir, folder_name)

            # Check if the item is a directory
            if os.path.isdir(folder_path):

                # Iterate over each file in the directory
                for file_name in os.listdir(folder_path):
                    
                    # Check if the file name ends with '_TRAIN.tsv'
                    if file_name.endswith('_TRAIN.csv'):
                        try:
                            # Print the file path
                            dataset_path = os.path.join(folder_path, file_name)
                            print(f"Starting to train {model_class.__name__} on {file_name}")
                            time.sleep(3)
                            train_model_on_dataset(model_class, dataset_path, param_list)
                        except Exception as e:
                            print(f" There was a problem training {file_name} on {model_class.__name__}, so it was skipped!")
                            print(e)
                            print()

                        
def train_model_on_dataset(model_class, dataset_path, param, batch_size=512, num_epochs=1000, 
                            criterion=nn.BCELoss(), optimizer=optim.Adam):
    # Load data from dataset_path
    classes_tensor, time_series_tensor = data_table_to_tensors(dataset_path, 'csv')

    # Initialize the model
    model = model_class(*param)

    # Define loss function and optimizer
    my_optimizer = optimizer(model.parameters())
    
    # Create DataLoader for training data
    my_dataset = TensorDataset(time_series_tensor, classes_tensor)
    my_data_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    trained_model = train_model(model, my_data_loader, criterion, my_optimizer, epochs=num_epochs)

    # Save the trained model
    model_path_trained = f"Models/{model_class.__name__}/{model_class.__name__}_{dataset_path.split('/')[-1]}"
    print(model_path_trained)
    torch.save(trained_model, model_path_trained)

    return model_path_trained


def example_train_model_on_dataset():
    """
    Example function to demonstrate usage of train_model_on_dataset.
    """
    param = [64,16]
    model_class = L2BinaryClassifier
    dataset_path = 'Example_Problems/Real_Data/UCRoutput/Adiac/Adiac_TRAIN.tsv'
    model_path = train_model_on_dataset(model_class, dataset_path, param)
    print("Trained model saved at:", model_path)

# Example usage when __name__ == '__main__':
if __name__ == '__main__':
    #example_train_model_on_dataset()
    train_all({L2BinaryClassifier:[64,32]})
