from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from Example_Problems.FordA import FordA_preprocessing
import torch

train_labels, train_inputs = FordA_preprocessing()
test_labels, test_inputs = FordA_preprocessing()


# Create the SVM model with RBF kernel
svm_model = SVC(kernel='rbf', C=0.98, gamma='scale', random_state=42)

# Train the model
svm_model.fit(train_inputs, train_labels)

# Prediction
predictions = svm_model.predict(test_inputs)

# Evaluate accuracy
accuracy = accuracy_score(torch.tensor(predictions), test_labels)
print(f"Accuracy: {accuracy}")
