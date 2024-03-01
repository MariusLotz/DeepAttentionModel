import torch
import os
from DeepAttentionModel.Functions.Helper_Functions import data_table_to_tensors
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Train_model import train_model
from Models.model_classes.L2BinaryClassifier import L2BinaryClassifier

#models = {L2BinaryClassifier:[p1,p2,p3], L2BinaryClassifier2:[p1,p2,p3], }

def make_folder(folder_name, path):
    folder_path = os.path.join(path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created successfully at '{path}'")
    else:
        print(f"Folder '{folder_name}' already exists at '{path}'")

def train_model_on_datasets(model_class, param_list, dataset_dir, trained_models_dir):
    for folder_name in os.listdir(dataset_dir):  # Iterate over each folder in the rdataset directory
        folder_path = os.path.join(dataset_dir, folder_name)
        if os.path.isdir(folder_path):  # Check if the item is a directory
            for file_name in os.listdir(folder_path):  # Iterate over each file in the directory
                if file_name.endswith('_TRAIN.csv'):
                    try:
                        dataset_path = os.path.join(folder_path, file_name)
                        print(f"Starting to train {model_class.__name__} on {file_name}")
                        train_model_on_dataset(model_class, dataset_path, param_list, trained_models_dir)
                    except Exception as e:
                        print(f" There was a problem training {file_name} on {model_class.__name__}, so it was skipped!")
                        print(e)


def train_model_on_dataset(model_class, dataset_path, param, trained_models_dir, batch_size=512, num_epochs=1000, 
                            criterion=nn.BCELoss(), optimizer=optim.Adam):
    
    classes_tensor, time_series_tensor = data_table_to_tensors(dataset_path, 'csv')
    model = model_class(*param)
    my_optimizer = optimizer(model.parameters())
    my_dataset = TensorDataset(time_series_tensor, classes_tensor)
    my_data_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
    trained_model = train_model(model, my_data_loader, criterion, my_optimizer, epochs=num_epochs)
    model_path_trained = f"{trained_models_dir}/{model_class.__name__}/{model_class.__name__}_{dataset_path.split('/')[-1]}"
    torch.save(trained_model, model_path_trained[:-4])  # remove .csv at the end


def train_models_on_datasets(models, dataset_dir, trained_models_dir="DeepAttentionModel/Models/trained_models"):
    for model_class, param_list in models.items():
        make_folder(model_class.__name__, trained_models_dir)  # create folder
        train_model_on_datasets(model_class, param_list, dataset_dir, trained_models_dir)  # train one model on all datasets
       

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
    train_models_on_datasets({L2BinaryClassifier:[64,32]},"DeepAttentionModel/Example_Problems/my_benchmark_dataset")
