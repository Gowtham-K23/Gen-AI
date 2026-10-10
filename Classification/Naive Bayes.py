# Naive Bayes Classification - Real World Dataset
# Dataset: Breast Cancer Wisconsin

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)

# Visualization 1: Feature distribution
plt.figure()
plt.hist(X[y == 0]['mean radius'], bins=20, alpha=0.7)
plt.hist(X[y == 1]['mean radius'], bins=20, alpha=0.7)
plt.xlabel("Mean Radius")
plt.ylabel("Frequency")
plt.title("Mean Radius Distribution by Class")
plt.legend(["Malignant", "Benign"])
plt.show()

# Visualization 2: Confusion Matrix
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.colorbar()
plt.show()
