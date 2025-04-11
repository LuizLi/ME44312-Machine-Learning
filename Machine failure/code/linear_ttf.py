import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. read the data
file_path = "data/data_with_ttf.csv"
df = pd.read_csv(file_path)

# 2. choose dependent and independent variable
X = df[['S13', 'S5', 'S16', 'S19', 'S18', 'S8', 'S15', 'S17']]
y = df['Time to Failure']

# 3. split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. build and train linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 6. predict the model
y_pred = model.predict(X_test_scaled)

# 7. evaulate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# 8. draw the outcome
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction")
plt.xlabel("True Time to Failure")
plt.ylabel("Predicted Time to Failure")
plt.title("Linear Regression: True vs Predicted TTF")
plt.legend()
plt.grid(True)
plt.show()
