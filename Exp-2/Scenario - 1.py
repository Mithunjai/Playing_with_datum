#Name:Mithunjai.E                                    24ADI003                                 24BAD071
# ===============================
# 1. Import Required Libraries
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

# ===============================
# 2. Load Dataset
# ===============================
bottle = "bottle.csv"
cast = "cast.csv"

b_df = pd.read_csv(bottle, low_memory=False)
c_df = pd.read_csv(cast, low_memory=False)

# ===============================
# 3. Merge Required Columns
# ===============================
merged = pd.merge(
    b_df,
    c_df[['Cst_Cnt', 'Sta_ID', 'Lat_Dec', 'Lon_Dec']],
    on=['Cst_Cnt', 'Sta_ID'],
    how='left'
)

# ===============================
# 4. Select Features & Target
# ===============================
features = ['Depthm', 'Salnty', 'O2ml_L', 'Lat_Dec', 'Lon_Dec']
target = 'T_degC'

# Remove missing values
merged = merged.dropna(subset=features + [target])

X = merged[features]
y = merged[target]

# ===============================
# 5. Feature Scaling
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 6. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=1
)

print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape :", X_test.shape, y_test.shape)

# ===============================
# 7. Linear Regression Model
# ===============================
lr = LinearRegression()
lr.fit(X_train, y_train)

# ===============================
# 8. Prediction
# ===============================
y_pred = lr.predict(X_test)

# ===============================
# 9. Model Evaluation
# ===============================
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Performance")
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)

# ===============================
# 10. Visualization
# ===============================

# Actual vs Predicted
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("Actual vs Predicted Water Temperature")
plt.show()

# Residual Errors
residuals = y_test - y_pred
plt.figure(figsize=(7,5))
sns.histplot(residuals, kde=True)
plt.xlabel("Residual Error")
plt.title("Residual Error Distribution")
plt.show()

# ===============================
# 11. Feature Selection
# ===============================
selector = SelectKBest(score_func=f_regression, k=3)
X_selected = selector.fit_transform(X_scaled, y)

selected_features = np.array(features)[selector.get_support()]
print("Selected Features:", selected_features)

# ===============================
# 12. Ridge Regression
# ===============================
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

print("\nRidge Regression R²:", r2_score(y_test, ridge_pred))

# ===============================
# 13. Lasso Regression
# ===============================
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

print("Lasso Regression R²:", r2_score(y_test, lasso_pred))

# ===============================
# 14. Feature Importance
# ===============================
coeff_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': lr.coef_
})

print("\nFeature Importance:")
print(coeff_df)
