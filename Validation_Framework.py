import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_binary_classifier(model, test_loader, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predictions = (outputs > threshold).float()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    # Convert lists to NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, conf_matrix

# Example Usage:
# Assuming you have a binary classifier model and a test_loader DataLoader for your test dataset
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# binary_classifier_model = ...  # Load your pre-trained binary classifier model here
# accuracy, precision, recall, f1, conf_matrix = evaluate_binary_classifier(binary_classifier_model, test_loader)
# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(f"Precision: {precision * 100:.2f}%")
# print(f"Recall: {recall * 100:.2f}%")
# print(f"F1 Score: {f1 * 100:.2f}%")
# print("Confusion Matrix:")
# print(conf_matrix)
