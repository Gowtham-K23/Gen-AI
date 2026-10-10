#1. Importing modules and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
    )

#2. Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns = data.feature_names)
y = data.target

print(X.shape)
print(np.unique(y))

#3. Exploratory data analysis
sns.countplot(x=y)
plt.title("Class Distribution (0 = Malignant, 1 = Benign)")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(X.corr(), cmap="coolwarm", linewidths=0.3)
plt.title("Feature Correlation Heatmap")
plt.show()

#4. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
    )

#6. Classification Models
#Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

#Decision Tree
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

#Random Forest
rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

#KNN
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

#Support vector machine
svm = SVC(kernel = 'rbf')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

#Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

#7. Classification Evaluation
def evaluate_model(name, y_test, y_pred):
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

evaluate_model("Naive Bayes", y_test, nb_pred)
evaluate_model("Decision Tree", y_test, dt_pred)
evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("KNN", y_test, knn_pred)
evaluate_model("SVM", y_test, svm_pred)
evaluate_model("Logistic Regression", y_test, lr_pred)

#8. Accuracy comparision - Classification model
models = ['Naive Bayes', 'Decision Tree', 'Random Forest', 'KNN', 'SVM', 'Logistic']
accuracies = [
    accuracy_score(y_test, nb_pred),
    accuracy_score(y_test, dt_pred),
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, knn_pred),
    accuracy_score(y_test, svm_pred),
    accuracy_score(y_test, lr_pred)
    ]

plt.bar(models, accuracies)
plt.ylabel("Accuracy")
plt.title("Classification Model Comparision")
plt.xticks(rotation = 45)
plt.show()

#9. Regression
y_reg = X['mean radius']
X_reg = X.drop(columns = ['mean radius'])

X_reg_scaled = scaler.fit_transform(X_reg)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg_scaled, y_reg, test_size = 0.2, random_state = 42)

lin_reg = LinearRegression()
lin_reg.fit(Xr_train, yr_train)
yr_pred = lin_reg.predict(Xr_test)

#10. Regression - Evaluation
print("MAE:", mean_absolute_error(yr_test, yr_pred))
print("MSE:", mean_squared_error(yr_test, yr_pred))
print("RMSE:", np.sqrt(mean_squared_error(yr_test, yr_pred)))
print("R2 Score:", r2_score(yr_test, yr_pred))

#11. Regression Visualization
plt.scatter(yr_test, yr_pred)
plt.xlabel("Actual Mean Radius")
plt.ylabel("Predicted Mean Radius")
plt.title("Actual vs Predicted (Linear Regression)")
plt.show()
