
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. read the data
file_path = "data/data_with_ttf.csv"  
df = pd.read_csv(file_path)

# 2. choose sensor variables and add non-linear features
sensor_features = ['S13', 'S5', 'S16', 'S19', 'S18', 'S8', 'S15', 'S17']

# choose dependent and independent variable and set non-linear variable
df['S13_squared'] = df['S13'] ** 2
df['S5_log'] = np.log1p(df['S5'].clip(lower=0))
df['S16_x_S18'] = df['S16'] * df['S18']
df['S19_sqrt'] = np.sqrt(df['S19'].clip(lower=0))
df['S8_squared'] = df['S8'] ** 2

# 3. combine original sensor + nonlinear features
X = df[sensor_features + ['S13_squared', 'S5_log', 'S16_x_S18', 'S19_sqrt', 'S8_squared']]
y = df['Time to Failure']

# 4. split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. build and train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 7. predict
y_pred = model.predict(X_test_scaled)

# 8. evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.4f}")

# 9. plot results
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction")
plt.xlabel("True Time to Failure")
plt.ylabel("Predicted Time to Failure")
plt.title("Linear Regression with Nonlinear Sensor Features")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
