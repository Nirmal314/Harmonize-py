import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.read_csv('tracks_features.csv')

# # print(df.columns)

# # # columns_to_drop = ['Unnamed: 0', 'artists', 'album_name', 'popularity', 'duration_ms', 'explicit', 'time_signature', 'track_genre', 'instrumentalness', 'liveness', 'speechiness']
# columns_to_drop = [
#     'album', 'album_id', 'artists', 'artist_ids',
#     'track_number', 'disc_number', 'explicit',
#     'duration_ms','time_signature', 'year', 'release_date'
# ]

# df = df.drop(columns=columns_to_drop)
# df = df.dropna()
# df = df.drop_duplicates()

# print(df.shape)
# df.to_csv('1.2mln.csv', index=False)

df = pd.read_csv('1.2mln.csv')

def handle_outliers(df, column, lower_percentile, upper_percentile):
    lower = df[column].quantile(lower_percentile)
    upper = df[column].quantile(upper_percentile)
    df[column] = df[column].clip(lower, upper)
    return df

df = handle_outliers(df, 'tempo', 0.01, 0.99)
df = handle_outliers(df, 'loudness', 0.01, 0.99)

def tempo_category(tempo):
    if tempo < 80:
        return 'slow'
    elif 80 <= tempo < 120:
        return 'medium'
    else:
        return 'fast'

def loudness_category(loudness):
    if loudness < -10:
        return 'quiet'
    elif -10 <= loudness < -5:
        return 'moderate'
    else:
        return 'loud'

def valence_category(valence):
    if valence < 0.33:
        return 'negative'
    elif 0.33 <= valence < 0.66:
        return 'neutral'
    else:
        return 'positive'

df['tempo_cat'] = df['tempo'].apply(tempo_category)
df['loudness_cat'] = df['loudness'].apply(loudness_category)
df['valence_cat'] = df['valence'].apply(valence_category)

features_to_normalize = ['valence', 'energy', 'danceability', 'acousticness', 'speechiness', 'instrumentalness', 'liveness']
for feature in features_to_normalize:
    df[feature] = df[feature].clip(0, 1)

mm_scaler = MinMaxScaler()
df['loudness_scaled'] = mm_scaler.fit_transform(df[['loudness']])

std_scaler = StandardScaler()
df['tempo_scaled'] = std_scaler.fit_transform(df[['tempo']])

features_for_pca = ['valence', 'energy', 'danceability', 'acousticness', 'loudness_scaled', 'tempo_scaled', 'speechiness', 'instrumentalness', 'liveness']

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[features_for_pca])

df['pca_1'] = pca_result[:, 0]
df['pca_2'] = pca_result[:, 1]

print("Explained variance ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(10, 8))
sns.scatterplot(x='pca_1', y='pca_2', data=df, alpha=0.1)
plt.title('PCA of Song Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(df[features_for_pca].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Audio Features')
plt.show()

df.to_csv('processed_tracks_features.csv', index=False)
