import pandas as pd
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# 加载数据
file_path = 'Machine failure/data/equipment_failure_data_1.csv'
data = pd.read_csv(file_path, sep=',')

# 查看数据概况
print("原始数据形状:", data.shape)
print("EQUIPMENT_FAILURE 分布:\n", data['EQUIPMENT_FAILURE'].value_counts())

# 处理非数值列（将类别变量编码为数值）
categorical_cols = ['REGION_CLUSTER', 'MAINTENANCE_VENDOR', 'MANUFACTURER', 'WELL_GROUP']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # 保存编码器，以便后续解码

# 将 DATE 转换为数值（从最早日期开始的天数）
data['DATE'] = pd.to_datetime(data['DATE'])
data['DATE'] = (data['DATE'] - data['DATE'].min()).dt.days

# 选择特征列（去掉 ID 和目标列）
feature_cols = [col for col in data.columns if col not in ['ID', 'EQUIPMENT_FAILURE']]
X = data[feature_cols]
y = data['EQUIPMENT_FAILURE']

# 应用 SMOTE，控制少数类数量
desired_minority_samples = 3000  # 可调整，例如 1000, 5000
smote = SMOTE(sampling_strategy={1: desired_minority_samples}, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 将结果转换回 DataFrame
resampled_data = pd.DataFrame(X_resampled, columns=feature_cols)
resampled_data['EQUIPMENT_FAILURE'] = y_resampled
resampled_data['ID'] = -1  # 新样本 ID 设为 -1

# 将编码的类别变量转换回原始值
for col in categorical_cols:
    resampled_data[col] = label_encoders[col].inverse_transform(resampled_data[col].astype(int))

# 将 DATE 转换回日期格式
resampled_data['DATE'] = pd.to_datetime(data['DATE'].min()) + pd.to_timedelta(resampled_data['DATE'], unit='D')

# 查看处理后的数据概况
print("SMOTE 后数据形状:", resampled_data.shape)
print("SMOTE 后 EQUIPMENT_FAILURE 分布:\n", resampled_data['EQUIPMENT_FAILURE'].value_counts())

# 保存结果
resampled_data.to_csv('Machine failure/data/equipment_failure_data_1_smote_all.csv', sep='\t', index=False)
print("结果已保存到 equipment_failure_data_1_smote_all.csv")