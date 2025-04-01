import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

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

    # 放宽预测正确的标准：预测值与真实值差值在 ±3 天范围内视为正确
    relaxed_accuracy = np.mean(np.abs(y_pred - y_test) <= 30)

    print(f"{model_name} Evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Relaxed Accuracy (±3 days): {relaxed_accuracy:.4f}")
    return y_pred

def evaluate_logistic_regression(model, X_test, y_test):
    """
    Evaluate the logistic regression model.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Logistic Regression Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)
    return y_pred

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
    evaluate_logistic_regression(logistic_model, X_test, y_test_clf)

    # 训练随机森林模型
    rf_model = train_random_forest(X_train, y_train_reg)
    # 评估随机森林模型
    evaluate_regression(rf_model, X_test, y_test_reg, "Random Forest")

    # 训练梯度提升模型
    gb_model = train_gradient_boosting(X_train, y_train_reg)
    # 评估梯度提升模型
    evaluate_regression(gb_model, X_test, y_test_reg, "Gradient Boosting")

if __name__ == "__main__":
    main()