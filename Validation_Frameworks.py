import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from Example_Problems.FordA import FordA_preprocessing
import numpy as np


def plot_confusion_matrix(conf_matrix, output_file="confusion_matrix1.png"):
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(output_file)  # Save the plot to a file
    plt.show()


def print_xth_to_yth(x,y, model, test_inputs, test_labels, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()  # Set the model to evaluation mode

    # Move inputs and labels to the specified device
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

    with torch.no_grad():
        outputs = model(test_inputs)
        predictions = torch.round(outputs)
    
    print("Values:")
    print(test_labels[x:y+1])
    print()
    print("Model predictions:")
    print(predictions[x:y+1])
    print()
    print(outputs[x:y+1])


def model_prediction(model, test_input, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    probability = model(test_input.torchno_grad)
    return torch.round(probability)


def model_predictions(model, test_inputs, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()  # Set the model to evaluation mode

    # Move inputs and labels to the specified device
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

    with torch.no_grad():
        probabilities= model(test_inputs)
        predictions = torch.round(probabilities)
    return predictions

#models = {'model1': model1, 'model2': model2}
def compare_models(**model, test_inputs, test_labels, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    modelnames_list = []
    metrics_list = []
    for model_name, model in models.items():
        modelnames_list.append(model_name)
        metric = model_metrics(model, test_inputs, test_labels)
        metrics_list.append(metric)


def model_metrics(model, test_inputs, test_labels, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    # Convert predictions and labels to numpy arrays
    y_pred = model_predictions(model, test_inputs).cpu().numpy()
    y_true = test_labels.cpu().numpy()

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Print metrics and confusion matrix
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot and save confusion matrix
    plot_confusion_matrix(conf_matrix, output_file="confusion_matrix.png")

    # Return metrics as a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }

    return metrics

def evaluate_model(model, test_inputs, test_labels, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    # Convert predictions and labels to numpy arrays
    y_pred = model_predictions(model, test_inputs).cpu().numpy()
    y_true = test_labels.cpu().numpy()

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Print metrics and confusion matrix
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot and save confusion matrix
    plot_confusion_matrix(conf_matrix, output_file="confusion_matrix.png")

    # Return metrics as a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }

    return metrics

if __name__ == "__main__":
    # Assuming you have a PyTorch model named 'model'
    labels, inputs = FordA_preprocessing(True)
    model = torch.load('Models/Feature2LBinaryClassifier_2024-01-30_trained')
    # Assuming you have test_inputs and test_labels as PyTorch tensors
    evaluate_model(model, inputs, labels, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    #print_xth_to_yth(0,10, model, inputs, labels, device=torch.device('cpu'))
