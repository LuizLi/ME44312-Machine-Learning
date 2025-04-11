import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取数据
def load_data(file_path):
    """
    Load the CSV data file.
    """
    df = pd.read_csv(file_path)
    return df

# 数据预处理
def preprocess_data(df):
    """
    Preprocess the data: select features, scale data, and split into train/test sets.
    """
    # 选择传感器数据作为特征
    sensor_columns = ['S15', 'S17', 'S13', 'S5', 'S16', 'S19', 'S18', 'S8']
    target_column = 'Time to Failure'

    # 特征和目标
    X = df[sensor_columns]
    y = df[target_column]

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 将目标变量分为两类（逻辑回归需要分类目标）
    y_classification = (y <= 30).astype(int)  # 30天内会坏为1，否则为0

    # 数据集划分
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X_scaled, y, y_classification, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf

# 线性回归模型
def train_linear_regression(X_train, y_train):
    """
    Train a linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# 逻辑回归模型
def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# 随机森林模型
def train_random_forest(X_train, y_train):
    """
    Train a random forest regression model.
    """
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

# 梯度提升模型
def train_gradient_boosting(X_train, y_train):
    """
    Train a gradient boosting regression model.
    """
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

# 模型评估
def evaluate_regression(model, X_test, y_test, model_name):
    """
    Evaluate regression models with relaxed accuracy criteria.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 放宽预测正确的标准：预测值与真实值差值在 ±30 天范围内视为正确
    relaxed_accuracy = np.mean(np.abs(y_pred - y_test) <= 30)

    print(f"{model_name} Evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Relaxed Accuracy (±30 days): {relaxed_accuracy:.4f}")

    # 对数据进行采样以减少点的数量
    sample_indices = np.arange(0, len(y_test), max(1, len(y_test) // 500))  # 每隔一定数量采样
    y_test_sampled = y_test.iloc[sample_indices]
    y_pred_sampled = y_pred[sample_indices]

    # 绘制实际值和预测值的对比图
    plt.figure(figsize=(10, 6))
    plt.plot(sample_indices, y_test_sampled, label="Actual Values", alpha=0.7, marker='o')
    plt.plot(sample_indices, y_pred_sampled, label="Predicted Values", alpha=0.7, marker='x')
    plt.title(f"{model_name}: Actual vs Predicted Values (Sampled)")
    plt.xlabel("Sample Index")
    plt.ylabel("Time to Failure")
    plt.legend()
    plt.grid(True)
    plt.show()

    return y_pred

def evaluate_logistic_regression(model, X_test, y_test, feature_names):
    """
    Evaluate the logistic regression model.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)  # Calculate AUC-ROC

    # Calculate relaxed accuracy: predictions within ±30 days
    relaxed_accuracy = np.mean(np.abs(y_pred - y_test) <= 30)

    print("Logistic Regression Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Relaxed Accuracy (±30 days): {relaxed_accuracy:.4f}")  # Print relaxed accuracy
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    # 输出特征的重要性（系数）
    print("\nFeature Importance (Logistic Regression):")
    for feature, coef in zip(feature_names, model.coef_[0]):
        print(f"{feature}: {coef:.4f}")

    # 可视化特征重要性
    plot_feature_importance(feature_names, model.coef_[0], "Logistic Regression")


    return y_pred

def evaluate_random_forest(model, X_test, y_test, feature_names, model_name="Random Forest"):
    """
    Evaluate the random forest regression model and output feature importance.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} Evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # 输出特征的重要性
    print("\nFeature Importance (Random Forest):")
    for feature, importance in zip(feature_names, model.feature_importances_):
        print(f"{feature}: {importance:.4f}")

    # 可视化特征重要性
    plot_feature_importance(feature_names, model.feature_importances_, model_name)

    return y_pred

def plot_feature_importance(feature_names, importances, model_name):
    """
    Plot feature importance as a horizontal bar chart with color indicating direction.
    """
    indices = np.argsort(importances)  # Sort features by importance (ascending)
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    # Set colors: green for positive, red for negative
    colors = ['green' if imp > 0 else 'red' for imp in sorted_importances]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_features)), sorted_importances, color=colors, alpha=0.7)
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.title(f"Feature Importance with Direction ({model_name})")
    plt.xlabel("Coefficient Value (Green=Positive, Red=Negative)")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 数据文件路径
    file_path = 'data/data_with_ttf.csv'

    # 加载数据
    df = load_data(file_path)

    # 数据预处理
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = preprocess_data(df)

    # 训练线性回归模型
    linear_model = train_linear_regression(X_train, y_train_reg)
    # 评估线性回归模型
    evaluate_regression(linear_model, X_test, y_test_reg, "Linear Regression")

    # 训练逻辑回归模型
    logistic_model = train_logistic_regression(X_train, y_train_clf)
    # 评估逻辑回归模型
    evaluate_logistic_regression(logistic_model, X_test, y_test_clf, feature_names=['S15', 'S17', 'S13', 'S5', 'S16', 'S19', 'S18', 'S8'])

    # 训练随机森林模型
    rf_model = train_random_forest(X_train, y_train_reg)
    # 评估随机森林模型
    evaluate_random_forest(rf_model, X_test, y_test_reg, feature_names=['S15', 'S17', 'S13', 'S5', 'S16', 'S19', 'S18', 'S8'])

    # 训练梯度提升模型
    gb_model = train_gradient_boosting(X_train, y_train_reg)
    # 评估梯度提升模型
    evaluate_regression(gb_model, X_test, y_test_reg, "Gradient Boosting")

if __name__ == "__main__":
    main()