from sklearn.datasets import make_blobs

# Generate synthetic data
X, y = make_blobs(n_samples=32**3, centers=15, n_features=3, random_state=42)

# Save to a CSV file
import pandas as pd
df = pd.DataFrame(X, columns=["Feature1", "Feature2", "Feature3"])
df.to_csv("synthetic/data.csv", index=False)