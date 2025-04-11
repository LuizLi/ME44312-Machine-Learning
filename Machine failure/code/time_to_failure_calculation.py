import pandas as pd

# Load data (replace with your actual file path)
df = pd.read_csv('Machine failure/data/equipment_failure_data_1.csv')  # Update with actual file name

# Sort by ID and AGE_OF_EQUIPMENT
df = df.sort_values(by=['ID', 'AGE_OF_EQUIPMENT'])

# Filter groups with at least one failure
df = df.groupby('ID').filter(lambda x: x['EQUIPMENT_FAILURE'].max() > 0)

# Compute Time to Failure
def compute_ttf_vectorized(df):
    # Get the age of the next failure per ID
    df['Next_Failure_Age'] = df.groupby('ID', group_keys=False).apply(
        lambda x: pd.to_numeric(x['AGE_OF_EQUIPMENT'].where(x['EQUIPMENT_FAILURE'] == 1).shift(-1).bfill(), downcast='integer'),
        include_groups=False
    ).fillna(float('inf'))  # Use inf for rows after last failure temporarily
    
    # Calculate TTF
    df['Time to Failure'] = df['Next_Failure_Age'] - df['AGE_OF_EQUIPMENT']
    
    # Set TTF to 0 for failure days
    df.loc[df['EQUIPMENT_FAILURE'] == 1, 'Time to Failure'] = 0
    
    # Filter out rows after the last failure per ID
    last_failure = df[df['EQUIPMENT_FAILURE'] == 1].groupby('ID')['AGE_OF_EQUIPMENT'].max()
    df = df.merge(last_failure.rename('Last_Failure_Age'), on='ID', how='left')
    df = df[df['AGE_OF_EQUIPMENT'] <= df['Last_Failure_Age']]
    
    # Drop temporary columns
    df = df.drop(columns=['Next_Failure_Age', 'Last_Failure_Age'])
    
    # Drop rows with NaN TTF (shouldn't happen now, but kept for safety)
    df = df.dropna(subset=['Time to Failure'])
    
    return df

# Apply computation
result_df = compute_ttf_vectorized(df)

# Display result
print(result_df[['ID', 'DATE', 'EQUIPMENT_FAILURE', 'AGE_OF_EQUIPMENT', 'Time to Failure']])

# Save to file
result_df.to_csv('Machine failure/data/data_with_ttf.csv', index=False)