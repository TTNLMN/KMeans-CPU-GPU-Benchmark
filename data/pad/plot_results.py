import pandas as pd
import matplotlib.pyplot as plt

labels = pd.read_csv('labels.csv')
data = pd.read_csv('data.csv')

# Filter out only the grey points
grey_data = data[data['Grey'] == 1][['X', 'Y']].reset_index(drop=True)

# Merge the filtered grey data with cluster labels
merged_data = pd.concat([grey_data, labels['ClusterID']], axis=1)

# Plotting
plt.figure(figsize=(10, 7))
for cluster in labels['ClusterID'].unique():
    cluster_data = merged_data[merged_data['ClusterID'] == cluster]
    plt.scatter(cluster_data['X'], cluster_data['Y'], label=f'Cluster {cluster}', s=20)

plt.title('Test Pad Clusters on PCB Image')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()