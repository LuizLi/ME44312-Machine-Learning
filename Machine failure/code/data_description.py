import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use('ggplot')  # Use ggplot style for clarity
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

# Load data
df = pd.read_csv('Machine failure/data/equipment_failure_data_1.csv')

# Basic data information
print("Basic Data Information:")
print(df.info())
print("\nMissing Value Statistics:")
print(df.isnull().sum())

# 1. ID - Unique identifier for specific machines
print("\nNumber of Unique Machines (ID):", df['ID'].nunique())

# 2. Categorical variables - REGION_CLUSTER, MAINTENANCE_VENDOR, MANUFACTURER, WELL_GROUP
categorical_cols = {
    'REGION_CLUSTER': 'Region of Machine',
    'MAINTENANCE_VENDOR': 'Maintenance Vendor',
    'MANUFACTURER': 'Manufacturer',
    'WELL_GROUP': 'Machine Type'
}
for col, label in categorical_cols.items():
    print(f"\n{label} ({col}) Distribution:")
    print(df[col].value_counts())
plt.figure(figsize=(12, 8))
for i, (col, label) in enumerate(categorical_cols.items(), 1):
    plt.subplot(2, 2, i)
    sns.countplot(data=df, x=col, palette='viridis')
    plt.title(f'{label} Distribution')
    plt.xlabel(label)
    plt.ylabel('Machine Count')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Machine failure/Graph/categorical_distribution.png')  # Save figure
plt.close()  # Close figure to free memory

# 3. Numeric variables - S15, S17, S13, S5, S16, S19, S18, S8, AGE_OF_EQUIPMENT
numeric_cols = {
    'S15': 'Sensor Value S15', 'S17': 'Sensor Value S17', 'S13': 'Sensor Value S13',
    'S5': 'Sensor Value S5', 'S16': 'Sensor Value S16', 'S19': 'Sensor Value S19',
    'S18': 'Sensor Value S18', 'S8': 'Sensor Value S8', 'AGE_OF_EQUIPMENT': 'Equipment Age (Days)'
}
for col, label in numeric_cols.items():
    print(f"\n{label} ({col}) Statistics:")
    print(df[col].describe())
plt.figure(figsize=(15, 10))
for i, (col, label) in enumerate(numeric_cols.items(), 1):
    plt.subplot(3, 3, i)
    df[col].hist(bins=30, color='skyblue', edgecolor='black')
    plt.title(f'{label} Distribution')
    plt.xlabel(label)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('Machine failure/Graph/numeric_distribution.png')  # Save figure
plt.close()  # Close figure to free memory

# 4. EQUIPMENT_FAILURE - Equipment failure status
print("\nEquipment Failure (EQUIPMENT_FAILURE) Distribution:")
failure_counts = df['EQUIPMENT_FAILURE'].value_counts(normalize=True)
print(f"No Failure (0): {failure_counts[0]:.2%}, Failure (1): {failure_counts[1]:.2%}")
plt.figure(figsize=(6, 5))
sns.countplot(data=df, x='EQUIPMENT_FAILURE', palette='Set2')
plt.title('Equipment Failure Distribution')
plt.xlabel('Failure Status (0: No Failure, 1: Failure)')
plt.ylabel('Observation Count')
plt.tight_layout()
plt.savefig('Machine failure/Graph/failure_distribution.png')  # Save figure
plt.close()  # Close figure to free memory