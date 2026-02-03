import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Dataset (FIXED)
# -----------------------------
df = pd.read_csv("marketing_campaign.csv", sep="\t")

# -----------------------------
# 2. Examine Structure & Nulls
# -----------------------------
print("First 5 rows:\n", df.head(), "\n")
print("Dataset Shape:", df.shape, "\n")

print("Data Info:")
df.info()

print("\nMissing Values:\n", df.isnull().sum())

print("\nSummary Statistics:\n", df.describe())

# -----------------------------
# 3. Feature Engineering
# -----------------------------
df['Age'] = 2024 - df['Year_Birth']

spending_cols = [
    'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]

df['Total_Spending'] = df[spending_cols].sum(axis=1)

# -----------------------------
# 4. Visualizations
# -----------------------------

# Age Distribution
plt.figure(figsize=(8,5))
df['Age'].value_counts().sort_index().plot(kind='bar')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=df['Age'])
plt.title("Age Box Plot")
plt.show()

# Income Distribution
plt.figure(figsize=(6,4))
sns.boxplot(x=df['Income'])
plt.title("Income Box Plot")
plt.show()

plt.figure(figsize=(8,5))
df['Income'].plot(kind='hist', bins=30)
plt.title("Income Histogram")
plt.xlabel("Income")
plt.show()

# Spending Patterns
plt.figure(figsize=(6,4))
sns.boxplot(x=df['Total_Spending'])
plt.title("Total Spending Box Plot")
plt.show()

plt.figure(figsize=(8,5))
df[spending_cols].mean().plot(kind='bar')
plt.title("Average Spending by Product Category")
plt.ylabel("Average Amount Spent")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 5. Insights Summary
# -----------------------------
print("\nINSIGHTS SUMMARY")
print("- Customers are mainly middle-aged, with some age outliers.")
print("- Income distribution is right-skewed with high-income outliers.")
print("- Spending varies significantly across customers.")
print("- Wine and meat products have the highest average spending.")
print("- High-income, high-spending customers are key marketing targets.")
