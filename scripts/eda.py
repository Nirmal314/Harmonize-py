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

# Summary statistics
print(df[features].describe())

# Plot histograms for each feature
df[features].hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Calculate correlation matrix
correlation_matrix = df[features].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Pair plot
sns.pairplot(df[features])
plt.show()

# Normalize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# # Convert to DataFrame for easier handling
scaled_df = pd.DataFrame(scaled_features, columns=features)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

# Convert PCA result to DataFrame
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

# # Plot PCA result
plt.figure(figsize=(10, 7))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5, c='blue')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Song Features')
plt.show()

# # Determine optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# ! result k = 3

# Apply K-Means with chosen number of clusters 
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to DataFrame
df['Cluster'] = clusters

# Visualize clusters in PCA space
plt.figure(figsize=(10, 7))
sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'], hue=df['Cluster'], palette='viridis', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clusters in PCA Space')
plt.legend(title='Cluster')
plt.show()
