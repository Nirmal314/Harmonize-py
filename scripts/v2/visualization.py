import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

df = pd.read_csv('processed_tracks_features.csv')

features_for_clustering = ['valence', 'energy', 'danceability', 'acousticness', 'loudness_scaled', 'tempo_scaled']

n_clusters = 5  
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(df[features_for_clustering])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[features_for_clustering])

df['pca_1'] = pca_result[:, 0]
df['pca_2'] = pca_result[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', palette='tab10', data=df, alpha=0.7)
plt.title('PCA of Song Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

cluster_centers = kmeans.cluster_centers_
centroid_df = pd.DataFrame(cluster_centers, columns=features_for_clustering)
print("Cluster centroids:")
print(centroid_df)

features_to_visualize = ['valence', 'energy', 'danceability', 'acousticness', 'loudness_scaled']

for feature in features_to_visualize:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y=feature, data=df)
    plt.title(f'Distribution of {feature} Across Clusters')
    plt.show()
    plt.savefig(f'images/{feature}_distribution.png')
    plt.close()

#! =======================================================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import euclidean_distances

# # Define the ideal points for each category based on feature characteristics
# category_ideal_points = {
#     'happy': [0.5, 0.7, 0.5, 0.3, 0.6, 0.6],
#     'sad': [0.2, 0.2, 0.3, 0.8, 0.2, -1.0],
#     'calm': [0.3, 0.3, 0.4, 0.9, 0.3, -0.5],
#     'confident': [0.6, 0.8, 0.6, 0.3, 0.7, 0.5],
#     'energetic': [0.6, 0.9, 0.7, 0.1, 0.8, 1.0]
# }

# # Convert to a DataFrame
# ideal_points_df = pd.DataFrame(category_ideal_points).T
# ideal_points_df.columns = ['valence', 'energy', 'danceability', 'acousticness', 'loudness_scaled', 'tempo_scaled']

# # Your cluster centroids
# cluster_centroids = pd.DataFrame({
#     'valence': [0.238813, 0.477870, 0.517740, 0.348496, 0.515578],
#     'energy': [0.193898, 0.605835, 0.672411, 0.258849, 0.724068],
#     'danceability': [0.352682, 0.432790, 0.570733, 0.473939, 0.573116],
#     'acousticness': [0.850548, 0.342744, 0.213697, 0.831560, 0.136472],
#     'loudness_scaled': [0.470625, 0.763481, 0.820674, 0.551090, 0.832148],
#     'tempo_scaled': [-1.200751, 1.724169, -0.726208, 0.201322, 0.427139]
# })

# # Compute distances from each centroid to each ideal point
# distances = pd.DataFrame(
#     euclidean_distances(cluster_centroids, ideal_points_df),
#     columns=ideal_points_df.index,
#     index=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
# )

# # Find the category with the minimum distance for each cluster
# assignments = distances.idxmin(axis=1)

# print(assignments)