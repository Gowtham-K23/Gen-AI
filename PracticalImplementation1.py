import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style = "whitegrid")

df = pd.read_csv("car_price_dataset_medium.csv")
print(df.head())

# Checking basic information
print("\n", df.shape, "\n") # Rows and columns
df.info() # Column types + missing values
df.describe() # statistics for numeric columns

# Check missing values
print(df.isnull().sum())

# Handle missing values
print("\nHandle missing values\n")

# Fill numerical missing values with mean
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Fill categorical missing values with mode
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

print("\n", df.isnull().sum())


# Histogram
df.hist(figsize = (12, 10))
plt.tight_layout()
plt.show()


# Boxplot to check outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include=['int64', 'float64']))
plt.xticks(rotation=90)
plt.show()


# Pairplot - relationship between features
sns.pairplot(df.select_dtypes(include = ['int64', 'float64']).iloc[:, :4])
plt.show()


df.to_csv("cleaned_data.csv", index = False)
