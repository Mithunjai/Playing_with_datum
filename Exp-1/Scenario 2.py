print("Mthunjai.E\t\tMachine learning Ex-1\t\t24BAD071")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_path = r"C:\Users\91638\Housing_Price_Prediction.csv"
data = pd.read_csv(data_path)

# Inspect dataset
print("\t\tData inspection:\n")
print("Using head on the dataset:\n")
print(data.head())

print("\nUsing tail on the dataset:\n")
print(data.tail())

# Inspect columns
print("\nInspecting columns of the dataset:\n")
print(data.columns)

print("\nUsing info on the dataset:\n")
data_info = data.info()
print(data_info)

print("\nUsing describe on the dataset:\n")
print(data.describe())

# Check missing values
print("\nMissing values on each column\n:")
missing_values = data.isnull().sum()
print(missing_values)

# Convert categorical columns to numeric (yes=1, no=0)
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
data[categorical_columns] = data[categorical_columns].apply(lambda x: x.map({'yes': 1, 'no': 0}))

# Visualizations
print("\t\tVisualizing correlations:\n")

# Scatter plot for 'area' vs 'price'
plt.figure(figsize=(12,8))
sns.scatterplot(x=data['area'], y=data['price'], color='skyblue')
plt.title("Area vs Price", fontsize=16)
plt.xlabel('Area', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.show()

# Heatmap for correlation
plt.figure(figsize=(12,8))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix Heatmap", fontsize=16)
plt.show()

# Scatter plot for 'bedrooms' vs 'price'
plt.figure(figsize=(12,6))
sns.scatterplot(x=data['bedrooms'], y=data['price'], color='lightgreen')
plt.title("Bedrooms vs Price", fontsize=16)
plt.xlabel('Bedrooms', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.show()
