from Functions.Helper_Functions import data_table_to_tensors
import torch
import os
from sklearn.metrics import accuracy_score
import pandas as pd
import re

def model_predictions(model, test_inputs, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        probabilities = model(test_inputs)
        predictions = torch.round(probabilities)
    return predictions

def get_paths_to_datasets(datasets_dir):
    paths = []
    for folder_name in os.listdir(datasets_dir):  # Iterate over each folder in the rdataset directory
        folder_path = os.path.join(datasets_dir, folder_name)
        if os.path.isdir(folder_path):  # Check if the item is a directory
            for file_name in os.listdir(folder_path):  # Iterate over each file in the directory
                if file_name.endswith('_TEST.csv'):
                     paths.append(os.path.join(folder_path, file_name))
    return paths

def get_paths_to_trained_models(models_dir, dataset_name):
    paths = []
    for folder_name in os.listdir(models_dir):  # Iterate over each folder in the rdataset directory
        folder_path = os.path.join(models_dir, folder_name)
        if os.path.isdir(folder_path):  # Check if the item is a directory
            for file_name in os.listdir(folder_path):  # Iterate over each file in the directory
                if dataset_name in file_name:
                     paths.append(os.path.join(folder_path, file_name))
    return paths

def validate_all_models_on_all_datasets(models_dir="Models/trained_models", datasets_dir="Example_Problems/my_benchmark_dataset", metric=accuracy_score, include_train_datasets=False):
    """Code needs adjusting in the future"""
    datasets_paths = get_paths_to_datasets(datasets_dir)
    pattern = r'/([^/]+)/[^/]+$'  # Regex to extract datasetname from datasetpath
    pattern_model_name = r'/([^/]+)_[^_]+_TRAIN$'  # Regex to extract model name
    my_dict = {}
    for dataset_path in datasets_paths:
        match = re.search(pattern, dataset_path)
        dataset_name = match.group(1)
        model_paths = get_paths_to_trained_models(models_dir, dataset_name)  # getting all model paths
        model_metric_scores = {}  # Initialize model_metric_scores as a dictionary
        for model_path in model_paths:
            #print(model_path)
            model = torch.load(model_path)
            model.eval()
            test_labels, test_inputs = data_table_to_tensors(dataset_path, 'csv')
            predictions = model_predictions(model, test_inputs)
            metric_value = metric(test_labels.cpu().numpy(), predictions.cpu().numpy())
            match_model_name = re.search(pattern_model_name, model_path)
            model_name = match_model_name.group(1)
            model_metric_scores[model_name] = metric_value  # Store metric score with model path as key
        my_dict[dataset_name] = model_metric_scores  # Assign model_metric_scores to corresponding dataset 
    df = pd.DataFrame(my_dict) 
    df = df.mean(axis=1)
    print(df)          

if __name__=="__main__":
    validate_all_models_on_all_datasets()

    