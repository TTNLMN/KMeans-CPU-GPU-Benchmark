from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# Load raw data
data = pd.read_csv('../data/raw/winequality-red.csv')

# PCA to reduce data dimensionality for plotting
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data.drop('quality', axis=1))
data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])

# Load processed data
labels = pd.read_csv('../data/processed/labels.csv')

# Merge data
data = pd.concat([data_pca, labels], axis=1)

# Plotting
plt.figure(figsize=(10, 7))
for cluster in data['ClusterID'].unique():
    cluster_data = data[data['ClusterID'] == cluster]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}', s=20)

plt.legend()
plt.title('Wine Groups')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
