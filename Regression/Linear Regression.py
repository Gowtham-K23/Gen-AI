import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load real-world dataset
data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

# Simple regression using one feature (Median Income)
X_simple = X[:, [feature_names.index("MedInc")]]

# Train-test split
X_train_s, X_test_s, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.3, random_state=42
)

X_train_m, X_test_m, _, _ = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Evaluation function
def evaluate(name, y_true, y_pred):
    print(name)
    print("MAE :", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R²  :", r2_score(y_true, y_pred))
    print("-" * 40)

# -----------------------------
# 1. Simple Linear Regression
# -----------------------------
simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train)
y_pred_simple = simple_model.predict(X_test_s)

evaluate("Simple Linear Regression", y_test, y_pred_simple)

plt.figure()
plt.scatter(X_test_s, y_test)
plt.scatter(X_test_s, y_pred_simple)
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("Simple Linear Regression")
plt.show()

# -----------------------------
# 2. Multiple Linear Regression
# -----------------------------
multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train)
y_pred_multi = multi_model.predict(X_test_m)

evaluate("Multiple Linear Regression", y_test, y_pred_multi)

# -----------------------------
# 3. Polynomial Regression
# -----------------------------
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_s)
X_test_poly = poly.transform(X_test_s)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

evaluate("Polynomial Regression (Degree 2)", y_test, y_pred_poly)

plt.figure()
plt.scatter(X_test_s, y_test)
plt.scatter(X_test_s, y_pred_poly)
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("Polynomial Regression")
plt.show()

# -----------------------------
# 4. Ridge Regression (L2)
# -----------------------------
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_m, y_train)
y_pred_ridge = ridge.predict(X_test_m)

evaluate("Ridge Regression (L2)", y_test, y_pred_ridge)

# -----------------------------
# 5. Lasso Regression (L1)
# -----------------------------
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_m, y_train)
y_pred_lasso = lasso.predict(X_test_m)

evaluate("Lasso Regression (L1)", y_test, y_pred_lasso)
