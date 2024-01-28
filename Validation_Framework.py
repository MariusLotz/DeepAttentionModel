import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from Example_Problems.FordA import FordA_preprocessing

def plot_confusion_matrix(conf_matrix, output_file="confusion_matrix1.png"):
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(output_file)  # Save the plot to a file
    plt.show()

def evaluate_model(model, test_inputs, test_labels, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Evaluate a PyTorch model on test inputs and labels.

    Parameters:
    - model (torch.nn.Module): Trained PyTorch model.
    - test_inputs (torch.Tensor): Test inputs tensor.
    - test_labels (torch.Tensor): Test labels tensor.
    - device (torch.device): Device on which to perform evaluation (default is 'cuda' if available, else 'cpu').

    Returns:
    - dict: A dictionary containing evaluation metrics.
    """
    model.eval()  # Set the model to evaluation mode

    # Move inputs and labels to the specified device
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

    with torch.no_grad():
        outputs = model(test_inputs)
        predictions = torch.round(torch.sigmoid(outputs))

    # Convert predictions and labels to numpy arrays
    y_pred = predictions.cpu().numpy()
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
    labels, inputs = FordA_preprocessing(False)
    model = torch.load('Models/Feature2LBinaryClassifier_2024-01-28_trained')
    # Assuming you have test_inputs and test_labels as PyTorch tensors
    evaluate_model(model, inputs, labels, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
