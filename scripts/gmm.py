import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('cleaned_dataset.csv')

df = pd.DataFrame(data)

# Select relevant features
features = ["danceability", "energy", "loudness", "speechiness", "acousticness", 
            "instrumentalness", "liveness", "valence", "tempo"]

# Assuming df is your dataframe with audio features
X = df[features]

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 1. Preprocessing: Select relevant features and scale them
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
            'instrumentalness', 'liveness', 'valence', 'tempo']

X = df[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
df['cluster'] = gmm.fit_predict(X_scaled)

# 3. Analyze Clusters: Calculate mean feature values for each cluster
cluster_centers = []
for cluster in range(3):
    cluster_centers.append(X[df['cluster'] == cluster].mean())

cluster_df = pd.DataFrame(cluster_centers, columns=features)

# 4. Assign mood labels based on cluster characteristics (heuristic approach)
# Here, you can assign the moods based on your analysis
# Example:
# - High valence + high energy --> HAPPY
# - Low valence + low energy --> SAD
# You can modify these based on your dataset and intuition.

cluster_to_mood = {
    0: 'HAPPY',
    1: 'SAD',
    2: 'ENERGETIC',
}

# Assign the mood label based on the cluster
df['mood'] = df['cluster'].map(cluster_to_mood)

# Display the dataframe with the mood labels
print(df[['danceability', 'energy', 'valence', 'tempo', 'cluster', 'mood']])

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce dimensions to 2D using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.title('Clusters in 2D PCA Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Plot histograms for selected features by cluster
features_to_plot = ['danceability', 'energy', 'valence']

plt.figure(figsize=(14, 8))

for i, feature in enumerate(features_to_plot):
    plt.subplot(2, 2, i+1)
    for cluster in range(7):
        plt.hist(df[df['cluster'] == cluster][feature], bins=30, alpha=0.5, label=f'Cluster {cluster}')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.show()
