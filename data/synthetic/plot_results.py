import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
data = pd.read_csv("data.csv")
labels = pd.read_csv("labels.csv")

# Combine data and labels
data['ClusterID'] = labels['ClusterID']

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot each cluster with a unique color
clusters = data['ClusterID'].unique()
for cluster in clusters:
    cluster_data = data[data['ClusterID'] == cluster]
    ax.scatter(cluster_data['Feature1'], 
               cluster_data['Feature2'], 
               cluster_data['Feature3'], 
               label=f'Cluster {cluster}', s=20)

# Customize the plot
ax.set_title("3D Scatter Plot of Clustering Results")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
ax.legend()

# Show the plot
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
data = pd.read_csv("data.csv")
labels = pd.read_csv("labels.csv")

# Combine data and labels
data['ClusterID'] = labels['ClusterID']

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot each cluster with a unique color
clusters = data['ClusterID'].unique()
for cluster in clusters:
    cluster_data = data[data['ClusterID'] == cluster]
    ax.scatter(cluster_data['Feature1'], 
               cluster_data['Feature2'], 
               cluster_data['Feature3'], 
               label=f'Cluster {cluster}', s=20)

# Customize the plot
ax.set_title("3D Scatter Plot of Clustering Results")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
ax.legend()

# Show the plot
plt.show()