from Helper_Functions import data_table_to_tensors
import torch
import os
from sklearn.metrics import accuracy_score


def model_predictions(model, test_inputs, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Make predictions for a batch of inputs using the provided model.

    Parameters:
    - model (torch.nn.Module): PyTorch model for predictions.
    - test_inputs (torch.Tensor): Batch of input data.
    - device (torch.device): Device for model computation.

    Returns:
    - torch.Tensor: Rounded predictions.
    """
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        probabilities = model(test_inputs)
        predictions = torch.round(probabilities)
    return predictions


def compare_models(models, test_inputs, test_labels, metric=accuracy_score):
    """
    Compare multiple models based on a specified metric.

    Parameters:
    - models (dict): Dictionary of model names and corresponding PyTorch models.
    - test_inputs (torch.Tensor): Batch of input data.
    - test_labels (torch.Tensor): Ground truth labels.
    - metric (function): Evaluation metric function.

    Returns:
    - tuple: Lists of model names and corresponding metric values.
    """
    model_name_list = []
    metric_list = []
    for model_name, model in models.items():
        model_name_list.append(model_name)
        predictions = model_predictions(model, test_inputs)
        metric_value = metric(test_labels.cpu().numpy(), predictions.cpu().numpy())
        metric_list.append(metric_value)
    return model_name_list, metric_list

def get_all_models_in_dir(directory):
    models = []
    for filename in os.listdir(directory):
        relative_path = os.path.join(directory, filename)
        if os.path.isfile(relative_path):
            model = torch.load(relative_path)
            model.eval()
            models.append(model)
    return models

def get_all_datasets(directory):
    datasets = []
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith("_TEST.csv"):
                    rel_file_path = os.path.join(folder_path, file)
                    classes_tensor, time_series_tensor = data_table_to_tensors(rel_file_path, 'csv')
                    datasets.append(classes_tensor, time_series_tensor)
    return datasets

def validate_and_compare(directory_models = "Models/L2BinaryClassifier", directory_data = "Example_Problems/Real_Data/UCRoutput"):  


    """
    classes_tensor, time_series_tensor = data_table_to_tensors(dataset_path, 'csv')

    models = torch.load(trained_model_path)
    model.eval()

    predicted_classes_tensor = model_predictions(model, time_series_tensor)
    """


if __name__=="__main__":
    main()

    