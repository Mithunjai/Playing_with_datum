print("Mthunjai.E\t\tMachine learning Ex-1\t\t24BAD071")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_path = r"C:\Users\91638\Hospital_Patient_Records.csv"
data = pd.read_csv(data_path)

# Inspect dataset
print("\t\tData inspection:\n")
print("Using head on the dataset:\n")
print(data.head())

print("\nUsing tail on the dataset:\n")
print(data.tail())

print("\nUsing info on the dataset:\n")
data_info = data.info()
print(data_info)

print("\nUsing describe on the dataset:\n")
print(data.describe())

# Check missing values
print("\nMissing values on each column\n:")
missing_values = data.isnull().sum()
print(missing_values)

# Visualizations
print("\t\tVisualizing distributions:\n")

# Histogram for glucose levels
plt.figure(figsize=(10,6))
sns.histplot(data['Glucose'], kde=True, color='skyblue')
plt.title("Glucose Level Distribution", fontsize=16)
plt.xlabel('Glucose Level', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# Boxplot for glucose levels
plt.figure(figsize=(8,6))
sns.boxplot(x=data['Glucose'], color='skyblue')
plt.title("Glucose Level Boxplot", fontsize=16)
plt.xlabel('Glucose Level', fontsize=12)
plt.show()

# Histogram for age distribution
plt.figure(figsize=(10,6))
sns.histplot(data['Age'], kde=True, color='lightgreen')
plt.title("Age Distribution", fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# Boxplot for age distribution
plt.figure(figsize=(8,6))
sns.boxplot(x=data['Age'], color='lightgreen')
plt.title("Age Boxplot", fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.show()

# Checking correlation between variables
correlation = data.corr()
print("\nCorrelation matrix:\n")
print(correlation)

# Heatmap for correlation
plt.figure(figsize=(12,8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix Heatmap", fontsize=16)
plt.show()
