from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# Load preprocessed data
data = pd.read_csv('../data/preprocessed_data.csv')

# Load labels
labels = pd.read_csv('labels.csv')

# Load centroids
centroids = pd.read_csv('centroids.csv')

# Create a DataFrame
data['ClusterID'] = labels['ClusterID']

# Plotting
plt.figure(figsize=(10, 7))
for cluster in data['ClusterID'].unique():
    cluster_data = data[data['ClusterID'] == cluster]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}', s=20)

plt.scatter(centroids['Dim0'], centroids['Dim1'], c='black', marker='X', s=100, label='Centroids')

plt.legend()
plt.title('Wine Groups')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
