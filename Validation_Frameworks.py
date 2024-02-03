import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from Example_Problems.FordA import FordA_preprocessing
import numpy as np


def plot_confusion_matrix(conf_matrix, output_file="confusion_matrix.png"):
    """
    Plots and saves the confusion matrix as a heatmap.

    Parameters:
    - conf_matrix (numpy.ndarray): Confusion matrix to be visualized.
    - output_file (str): File path to save the confusion matrix plot.
    """
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(output_file)  # Save the plot to a file
    plt.show()


def model_prediction(model, test_input, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Make a single prediction using the provided model.

    Parameters:
    - model (torch.nn.Module): PyTorch model for prediction.
    - test_input (torch.Tensor): Input data for making a prediction.
    - device (torch.device): Device for model computation.

    Returns:
    - torch.Tensor: Rounded prediction.
    """
    model.eval()
    with torch.no_grad():
        probability = model(test_input)
    return torch.round(probability)


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


def print_compared_models(model_name_list, metric_list):
    """
    Print the compared models and their corresponding metric values.

    Parameters:
    - model_name_list (list): List of model names.
    - metric_list (list): List of metric values.
    """
    for model_name, metric_value in zip(model_name_list, metric_list):
        print(f'{model_name}: {metric_value}')


def confusion_matrix(model, test_inputs, test_labels):
    """
    Evaluate a model's performance using metrics and plot the confusion matrix.

    Parameters:
    - model (torch.nn.Module): PyTorch model to be evaluated.
    - test_inputs (torch.Tensor): Batch of input data.
    - test_labels (torch.Tensor): Ground truth labels.

    Returns:
    - dict: Dictionary containing evaluation metrics.
    """
    y_pred = model_predictions(model, test_inputs).cpu().numpy()
    y_true = test_labels.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    conf_matrix = confusion_matrix(y_true, y_pred)

    print(conf_matrix)

    plot_confusion_matrix(conf_matrix, output_file="confusion_matrix.png")

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def Test_framework():
    labels, inputs = FordA_preprocessing(True)
    models = {'Feature2LBinaryClassifier': torch.load('Models/Feature2LBinaryClassifier_2024-02-03_trained'),
            'SimpleBinaryClassifier': torch.load('Models/SimpleBinaryClassifier_2024-01-30_trained'),
            'RawSimpleBinaryClassifier':torch.load('Models/RawSimpleBinaryClassifier_2024-02-01_trained')}

    model_name_list, metric_list = compare_models(models, inputs, labels)
    print_compared_models(model_name_list, metric_list)


if __name__ == "__main__":
    Test_framework()
    
