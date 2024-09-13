import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load data
data = pd.read_csv('cleaned_dataset.csv')

df = pd.DataFrame(data)

# Select relevant features for clustering
features = ['valence', 'energy', 'danceability', 'acousticness', 'loudness', 'tempo', 'speechiness', 'instrumentalness', 'liveness']

# Standardize the features (Z-score normalization)
scaler = StandardScaler()
standardized_data = scaler.fit_transform(df[features])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=7, random_state=42)
labels = kmeans.fit_predict(standardized_data)
df['cluster'] = labels

# Calculate summary statistics (mean, median, std) for each cluster
cluster_summary = df.groupby('cluster')[features].agg(['mean', 'median', 'std'])
print("Cluster Summary (Mean, Median, Std):\n", cluster_summary)

# Calculate overall means and standard deviations of the dataset
overall_mean = df[features].mean()
overall_std = df[features].std()

# Function to assign mood labels based on z-scores (standardized deviation from mean)
def assign_mood(cluster_means, cluster_std):
    z_valence = (cluster_means['valence', 'mean'] - overall_mean['valence']) / overall_std['valence']
    z_energy = (cluster_means['energy', 'mean'] - overall_mean['energy']) / overall_std['energy']
    z_acousticness = (cluster_means['acousticness', 'mean'] - overall_mean['acousticness']) / overall_std['acousticness']
    z_danceability = (cluster_means['danceability', 'mean'] - overall_mean['danceability']) / overall_std['danceability']
    z_loudness = (cluster_means['loudness', 'mean'] - overall_mean['loudness']) / overall_std['loudness']
    z_tempo = (cluster_means['tempo', 'mean'] - overall_mean['tempo']) / overall_std['tempo']

    # Mood assignment based on z-scores
    if z_valence > 0.6 and z_energy > 0.6 and z_danceability > 0.6:
        return 'HAPPY'
    elif z_valence < -0.5 and z_energy < -0.5:
        return 'SAD'
    elif z_acousticness > 0.5 and z_energy < -0.4:
        return 'RELAXING'
    elif z_energy > 0.7 and z_tempo > 0.6 and z_danceability > 0.6:
        return 'ENERGETIC'
    elif z_loudness > 0.5 and z_energy > 0.5 and z_valence > 0.4:
        return 'CONFIDENT'
    elif z_valence < -0.6 and z_loudness > 0.4 and df['speechiness'].mean() > 0.4:
        return 'SCARY'
    elif z_valence > 0.5 and z_energy < 0.5 and z_acousticness > 0.5:
        return 'CALM'
    else:
        return 'NEUTRAL'  # Default if no condition matches precisely

# Assign mood labels to clusters
cluster_moods = {}
for cluster_id in cluster_summary.index:
    cluster_means = cluster_summary.loc[cluster_id]  # Use means for assignment
    mood = assign_mood(cluster_means, None)
    cluster_moods[cluster_id] = mood

# Map clusters to moods
df['mood'] = df['cluster'].map(cluster_moods)

# Output the first few rows of the dataframe with the assigned mood
print(df[['track_name', 'artists', 'mood']])

# Save results to a CSV file
df.to_csv('results_with_moods.csv', index=False)

# Optional: Visualize clusters using PCA for 2D visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(standardized_data)
df['pca_1'] = pca_result[:, 0]
df['pca_2'] = pca_result[:, 1]

# Plot the clusters and assigned moods
plt.figure(figsize=(10, 7))
sns.scatterplot(x='pca_1', y='pca_2', hue='mood', data=df, palette='Set1')
plt.title('Clusters Visualization (PCA-reduced)')
plt.show()
