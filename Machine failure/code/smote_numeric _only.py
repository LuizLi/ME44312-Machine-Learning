import pandas as pd
from imblearn.over_sampling import SMOTE

# 加载数据
file_path = 'Machine failure/data/equipment_failure_data_1.csv' 
data = pd.read_csv(file_path, sep=',')  

# 检查加载后的数据
print("数据预览:\n", data.head())
print("列名:", data.columns.tolist())
print("原始数据形状:", data.shape)

# 确认目标列名是否存在（假设为 'EQUIPMENT_FAILURE'）
target_col = 'EQUIPMENT_FAILURE'
if target_col not in data.columns:
    raise ValueError(f"列 '{target_col}' 不存在，请检查文件列名。")

# 查看目标列分布
print(f"{target_col} 分布:\n", data[target_col].value_counts())

# 定义数值特征
numeric_cols = ['S15', 'S17', 'S13', 'S5', 'S16', 'S19', 'S18', 'AGE_OF_EQUIPMENT']

# 检查数值列是否存在
missing_cols = [col for col in numeric_cols if col not in data.columns]
if missing_cols:
    raise ValueError(f"以下数值列缺失: {missing_cols}")

# 分离数值特征和目标
X_numeric = data[numeric_cols]
y = data[target_col]

# 应用 SMOTE，控制少数类数量
desired_minority_samples = 3000
smote = SMOTE(sampling_strategy={1: desired_minority_samples}, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_numeric, y)

# 将结果转换为 DataFrame，仅保留数值特征和目标列
resampled_data = pd.DataFrame(X_resampled, columns=numeric_cols)
resampled_data[target_col] = y_resampled

# 查看处理后的数据概况
print("SMOTE 后数据形状:", resampled_data.shape)
print(f"SMOTE 后 {target_col} 分布:\n", resampled_data[target_col].value_counts())

# 保存结果
resampled_data.to_csv('Machine failure/data/equipment_failure_data_1_smote_numeric_only.csv', sep='\t', index=False)
print("结果已保存到 equipment_failure_data_1_smote_numeric_only.csv")