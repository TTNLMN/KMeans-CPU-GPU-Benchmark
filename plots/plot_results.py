from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# Load standardized data
data = pd.read_csv('../data/preprocessed_data.csv')

# Load labels
labels = pd.read_csv('cluster_labels.csv')

# Perform PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)

# Create a DataFrame
df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
df['ClusterID'] = labels['ClusterID']

# Plotting
plt.figure(figsize=(10, 7))
for cluster in df['ClusterID'].unique():
    cluster_data = df[df['ClusterID'] == cluster]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}', s=20)

plt.legend()
plt.title('Wine Groups')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
