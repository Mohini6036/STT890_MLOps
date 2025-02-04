import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

data_path = r'D:\Spring_2025\STT_890\STT890_MLOps\data\sampregdata.csv'
df = pd.read_csv(data_path)

correlations = df.corr()['y'].drop('y')
best_X = correlations.abs().idxmax()

print(f"Best predictor for Y: {best_X}")

X = df[[best_X]]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_1 = LinearRegression()
model_1.fit(X_train, y_train)

y_pred_1 = model_1.predict(X_test)
mse_1 = mean_squared_error(y_test, y_pred_1)
r2_1 = r2_score(y_test, y_pred_1)

print(f"Model 1 - MSE: {mse_1}, R2: {r2_1}")

os.makedirs("models", exist_ok=True)
joblib.dump(model_1, "models/model_1.pkl")

second_best_X = correlations.abs().nlargest(2).index[1]
print(f"Second predictor for Model 2: {second_best_X}")

X2 = df[[best_X, second_best_X]]
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.3, random_state=42)

model_2 = LinearRegression()
model_2.fit(X2_train, y_train)

y_pred_2 = model_2.predict(X2_test)
mse_2 = mean_squared_error(y_test, y_pred_2)
r2_2 = r2_score(y_test, y_pred_2)

print(f"Model 2 - MSE: {mse_2}, R2: {r2_2}")

joblib.dump(model_2, "models/model_2.pkl")

readme_content = f"""
# Regression Models

## Model 1 (Single Predictor: {best_X})
- MSE: {mse_1}
- R2 Score: {r2_1}

## Model 2 (Two Predictors: {best_X}, {second_best_X})
- MSE: {mse_2}
- R2 Score: {r2_2}

### Comparison
Model 2, which uses two predictors, performs better than Model 1 if R2 is higher and MSE is lower. 
"""

with open("README.md", "w") as f:
    f.write(readme_content)

print("README file created successfully.")