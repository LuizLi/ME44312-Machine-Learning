import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

def plot_conditions(df, failure_df, parameters):
    for parameter in parameters:
        plt.figure(figsize=(10, 6))
        plt.scatter(df.index, df[parameter], label=f'{parameter} (Normal)', color='blue', s=10, alpha=0.5)
        plt.scatter(failure_df.index, failure_df[parameter], label=f'{parameter} (Failure)', color='red', s=10)
        plt.xlabel('Instance')
        plt.ylabel(parameter)
        plt.title(f'{parameter} Levels with Equipment Failures Highlighted')
        plt.legend()
        plt.tight_layout()
        plt.show()

def predict_failures(df, features):
    X = df[features]
    y = df['EQUIPMENT_FAILURE']

    # Split the data into training and testing sets using stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def analyze_failure_patterns(df, features, n_clusters=3):
    # Filter the DataFrame to include only rows where EQUIPMENT_FAILURE == 1
    failure_df = df[df['EQUIPMENT_FAILURE'] == 1]

    # Select features (sensor data) for clustering
    X = failure_df[features]

    # Initialize and fit the K-means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # Add cluster labels to the DataFrame
    failure_df['Cluster'] = kmeans.labels_

    # Print cluster centers
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    print("Cluster Centers:")
    print(cluster_centers)

    # Plot radar chart for each cluster center
    labels = features
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Initialize the radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Plot each cluster center
    for i, row in cluster_centers.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.25)

    # Add labels and title
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title('Cluster Centers Radar Chart')
    plt.legend()
    plt.show()

def analyze_failure_patterns_pca_kmeans(df, features, n_clusters=3, n_components=2):
    # Filter the DataFrame to include only rows where EQUIPMENT_FAILURE == 1
    failure_df = df[df['EQUIPMENT_FAILURE'] == 1]

    # Select features (sensor data) for clustering
    X = failure_df[features]

    # Use PCA to reduce dimensions
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Initialize and fit the K-means model on PCA-reduced data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_pca)

    # Add cluster labels to the DataFrame
    failure_df['Cluster'] = kmeans.labels_

    # Generate dynamic column names for PCA components
    pca_columns = [f'PCA Component {i+1}' for i in range(n_components)]

    # Print cluster centers
    print("PCA-K-means Cluster Centers:")
    print(pd.DataFrame(kmeans.cluster_centers_, columns=pca_columns))

    # Plot the clusters in 2D if n_components is 2
    if n_components == 2:
        plt.figure(figsize=(10, 6))
        for cluster in range(n_clusters):
            cluster_data = X_pca[failure_df['Cluster'] == cluster]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}', s=10)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('PCA-K-means Clustering of Equipment Failure Data')
        plt.legend()
        plt.tight_layout()
        plt.show()

def find_optimal_k(df, features, k_range=range(2, 11)):
    # Filter the DataFrame to include only rows where EQUIPMENT_FAILURE == 1
    failure_df = df[df['EQUIPMENT_FAILURE'] == 1]

    # Select features (sensor data) for clustering
    X = failure_df[features]

    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    # Plot SSE for each k
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal k')
    plt.show()

def main():
    # Define the path to the CSV file
    csv_file_path = os.path.join('data', 'equipment_failure_data_1.csv')

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Filter the DataFrame to include only rows where EQUIPMENT_FAILURE == 1
    failure_df = df[df['EQUIPMENT_FAILURE'] == 1]

    # Plot the conditions for all data points, highlighting failures
    parameters = [column for column in df.columns if column.startswith('S')]
    plot_conditions(df, failure_df, parameters)

    # Select features (sensor data) and target (EQUIPMENT_FAILURE)
    features = [column for column in df.columns if column.startswith('S')]
    predict_failures(df, features)

    # Find the optimal number of clusters using the Elbow Method
    find_optimal_k(df, features)

    # Analyze failure patterns using K-means clustering with the chosen number of clusters
    optimal_k = 5  # Replace this with the chosen k value based on the Elbow Method plot
    analyze_failure_patterns(df, features, n_clusters=optimal_k)

    # Analyze failure patterns using PCA-K-means clustering with the chosen number of clusters
    analyze_failure_patterns_pca_kmeans(df, features, n_clusters=optimal_k)

if __name__ == "__main__":
    main()