import pandas as pd
import matplotlib.pyplot as plt

# Load data with (X, Y) and cluster labels
data = pd.read_csv('../data/processed/labels.csv')
raw_data = pd.read_csv('../data/raw/test_pad.csv')

# Filter out only the grey points
grey_data = raw_data[raw_data['Grey'] == 1][['X', 'Y']].reset_index(drop=True)

# Merge the filtered grey data with cluster labels
data = pd.concat([grey_data, data['ClusterID']], axis=1)

# Plotting
plt.figure(figsize=(10, 7))
for cluster in data['ClusterID'].unique():
    cluster_data = data[data['ClusterID'] == cluster]
    plt.scatter(cluster_data['X'], cluster_data['Y'], label=f'Cluster {cluster}', s=5)

plt.title('Test Pad Clusters on PCB Image')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
