import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('equipment_failure_data_1.csv')
df = df.drop(['DATE', 'ID'], axis=1)

# Feature columns
categorical_cols = ['REGION_CLUSTER', 'MAINTENANCE_VENDOR', 'MANUFACTURER', 'WELL_GROUP']
numerical_cols = ['S15', 'S17', 'S13', 'S5', 'S16', 'S19', 'S18', 'S8', 'AGE_OF_EQUIPMENT']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train-test split
X = df.drop('EQUIPMENT_FAILURE', axis=1)
y = df['EQUIPMENT_FAILURE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# ======= Evaluation =======
metrics = {
    "R² Score": r2_score(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred)
}

print("\n" + "="*45)
print(f"{'Model Evaluation Metrics':^45}")
print("-"*45)
for name, value in metrics.items():
    print(f"{name + ':':<12} {value:.4f}")
print("="*45)

# ======= Feature Importance Plot =======

# Rename column for display
num_features = [col if col != 'AGE_OF_EQUIPMENT' else 'OF_EQUIPMENT' for col in numerical_cols]

# Get coefficients
coefs = model.named_steps['regressor'].coef_
num_cols_indices = preprocessor.transformers_[0][2]
numeric_coefs = coefs[:len(num_cols_indices)]

# Create sorted DataFrame
importance_df = pd.DataFrame({
    'Feature': num_features,
    'Coefficient': numeric_coefs
}).sort_values(by='Coefficient', key=abs, ascending=False)

# Plot setup
plt.figure(figsize=(10, 6))

# Color by sign
colors = ['#2ecc71' if coef > 0 else '#e74c3c' for coef in importance_df['Coefficient']]

# Draw bars
bars = plt.barh(importance_df['Feature'], 
               importance_df['Coefficient'], 
               color=colors,
               height=0.7,
               edgecolor='black')

# Adjust x-axis
max_coef = importance_df['Coefficient'].abs().max()
plt.xlim(-max_coef*1.3, max_coef*1.3)

# Add value labels
for bar, color in zip(bars, colors):
    width = bar.get_width()
    label_pos = width + (0.02 if width > 0 else -0.02)*max_coef
    plt.text(label_pos, 
            bar.get_y() + bar.get_height()/2,
            f"{width:.4f}",
            va='center',
            ha='left' if width > 0 else 'right',
            color=color)

# Labels and title
plt.xlabel('Coefficient Value', fontsize=12, labelpad=10)
plt.ylabel('Sensor Features', fontsize=12, labelpad=10)
plt.title('Feature Importance Direction Analysis', fontsize=14, pad=20)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle=':', alpha=0.5)

# Summary annotation
plt.text(0.95, 0.05,
        f"Avg Absolute Impact: {importance_df['Coefficient'].abs().mean():.4f}",
        transform=plt.gca().transAxes,
        ha='right',
        bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()
