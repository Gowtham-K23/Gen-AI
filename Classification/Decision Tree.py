import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#1. Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='Species')

#Map target numbers to species names for clarity
y = y.map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("First 5 rows of features:")
print(X.head())
print("\nTarget classes:")
print(y.value_counts())

#2. Visualize
sns.pairplot(pd.concat([X, y], axis=1), hue='Species', palette='Set2')
plt.suptitle("Iris Dataset Pairplot", y=1.02)
plt.show()

#3. Training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#4. Train decision tree classifier
dt_model = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state = 42)
dt_model.fit(X_train, y_train)

#5. Make predictions
y_pred = dt_model.predict(X_test)

#6. Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#7. Visualize the decision tree
plt.figure(figsize=(12,8))
plot_tree(dt_model, 
          feature_names=X.columns, 
          class_names=iris.target_names, 
          filled=True, rounded=True, fontsize=12)
plt.title("Decision Tree Visualization")
plt.show()
