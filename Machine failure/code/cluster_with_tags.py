import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def preprocess_and_standardize(file_path):
    """
    读取数据文件，对所有传感器数据进行标准化，并对分类变量进行 Label Encoding。
    同时输出分类变量的每种标签对应的标准化数值。
    """
    from sklearn.preprocessing import LabelEncoder

    # 读取数据
    df = pd.read_csv(file_path)

    # 筛选出传感器列（以 'S' 开头的列）
    sensor_columns = [col for col in df.columns if col.startswith('S')]

    # 筛选出分类变量列
    categorical_columns = ['REGION_CLUSTER', 'MAINTENANCE_VENDOR', 'MANUFACTURER', 'WELL_GROUP']

    # 对分类变量进行 Label Encoding
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # 保存编码器以备后续使用

        # 输出每个分类变量的 Label Encoding 结果
        print(f"Label Encoding for {col}:")
        for original, encoded in zip(le.classes_, range(len(le.classes_))):
            print(f"  {original} -> {encoded}")

    # 筛选出 EQUIPMENT_FAILURE == 1 的数据
    failure_df = df[df['EQUIPMENT_FAILURE'] == 1]

    # 合并传感器列和分类变量列
    all_features = sensor_columns + categorical_columns
    # print("All features selected for standardization and clustering:")
    # print(failure_df[all_features].head())

    # 对所有特征列进行标准化
    scaler = StandardScaler()
    failure_df[all_features] = scaler.fit_transform(failure_df[all_features])

    # print("All features selected for standardization and clustering:")
    # print(failure_df[all_features].head())

    
    # 输出分类变量的每种标签对应的标准化数值
    print("\nStandardized values for categorical variables:")
    for col in categorical_columns:
        print(f"\n{col}:")
        le = label_encoders[col]
        col_index = all_features.index(col)  # 找到分类变量在 all_features 中的索引
        mean = scaler.mean_[col_index]      # 获取分类变量的均值
        std = scaler.scale_[col_index]      # 获取分类变量的标准差

        # 使用完整的 Label Encoding 范围计算标准化值
        for original, encoded in zip(le.classes_, range(len(le.classes_))):
            standardized_value = (encoded - mean) / std
            print(f"  {original} -> {standardized_value:.4f}")

    return failure_df, all_features

def find_optimal_k(data, features, k_range=range(2, 15)):
    """
    使用肘部法（Elbow Method）选择最优的 k 值。
    """
    X = data[features]
    sse = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    # 绘制肘部法图表
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal k')
    plt.show()

def perform_kmeans_clustering(data, features, n_clusters):
    """
    对数据进行 K-means 聚类，并返回聚类后的数据和聚类中心。
    """
    X = data[features]

    # 进行 K-means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X)  # 将聚类标签添加到原始数据中

    # 获取聚类中心
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)

    return data, cluster_centers

def save_cluster_data_to_excel(data, n_clusters, output_file="cluster_results_with_tags.xlsx"):
    """
    将每个聚类的数据保存到单个 Excel 文件的不同工作表中，每个工作表只包含指定的列。
    """
    with pd.ExcelWriter(output_file) as writer:
        for cluster in range(n_clusters):
            cluster_data = data[data['Cluster'] == cluster]
            sheet_name = f"Cluster_{cluster}"
            
            # # 打印输出 cluster_data
            # print(f"Cluster {cluster} data:")
            # print(cluster_data[['ID', 'REGION_CLUSTER', 'MAINTENANCE_VENDOR', 'MANUFACTURER', 'WELL_GROUP', 'AGE_OF_EQUIPMENT', 'S15', 'S17', 'S13', 'S5', 'S16', 'S19', 'S18', 'S8']])
            
            # 保存到 Excel
            cluster_data[['ID', 'REGION_CLUSTER', 'MAINTENANCE_VENDOR', 'MANUFACTURER', 'WELL_GROUP', 'AGE_OF_EQUIPMENT', 'S15', 'S17', 'S13', 'S5', 'S16', 'S19', 'S18', 'S8']].to_excel(
                writer, sheet_name=sheet_name, index=False
            )

def plot_radar_chart(cluster_centers, features):
    """
    绘制雷达图，展示每个聚类中心的参数水平。
    """
    labels = features
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # 设置颜色和透明度
    colors = plt.cm.get_cmap('tab10', len(cluster_centers))  # 使用 'tab10' 配色方案

    for i, row in cluster_centers.iterrows():
        values = row.tolist()
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, label=f'Cluster {i}', linewidth=2)  # 增加线条宽度
        ax.fill(angles, values, color=colors(i), alpha=0.3)  # 调整透明度

    # 添加径向网格线
    ax.yaxis.grid(color="gray", linestyle="dotted", alpha=0.7)

    # 设置标签和标题
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    plt.title('Sensor Radar Chart for different clusters', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))  # 调整图例位置
    plt.tight_layout()
    plt.show()

def main():
    # 数据文件路径
    file_path = 'data/equipment_failure_data_1.csv'

    # 数据预处理和标准化
    failure_df, features = preprocess_and_standardize(file_path)

    # 使用肘部法选择最优 k 值
    find_optimal_k(failure_df, features)

    # 根据肘部法选择的最优 k 值进行聚类分析
    optimal_k = 7  # 根据肘部法图表选择的 k 值
    clustered_data, cluster_centers = perform_kmeans_clustering(failure_df, features, n_clusters=optimal_k)

    # 保存每个聚类的数据到单个 Excel 文件的不同工作表中
    save_cluster_data_to_excel(clustered_data, n_clusters=optimal_k, output_file="cluster_results_with_tags.xlsx")

    # 绘制雷达图
    plot_radar_chart(cluster_centers, features)

if __name__ == "__main__":
    main()