import spotipy
from dotenv import load_dotenv
import os
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

load_dotenv()

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri='https://google.com'
                                               ))
                                               

playlist_id = '2uORYX3pVmRBUJe8uXrK8H'
results = sp.playlist_tracks(playlist_id)
tracks = results['items']

# Extract features and URLs
data = []
urls = []  

for track in tracks:
    song_id = track['track']['id']
    features = sp.audio_features(song_id)[0]
    if features is not None:
        data.append(features)
        urls.append(track['track']['external_urls']['spotify'])

print(data)

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Observe correletion

# # print(df[['danceability', 'energy','loudness', 'speechiness', 'acousticness', 'instrumentalness', 'tempo', 'valence']].corr())

# # Drop non-numeric columns if any
# df = df.select_dtypes(include=[float, int])

# # Add URLs to DataFrame
# df['song_url'] = urls

# # Feature engineering
# df['harmonic_complexity'] = df['key'] * df['mode']
# df['rhythmic_variability'] = df['tempo'].rolling(window=5).std()

# # Handle missing values (NaNs) using SimpleImputer
# numeric_features = df.select_dtypes(include=[float, int]).copy()
# imputer = SimpleImputer(strategy='mean')
# numeric_features_imputed = imputer.fit_transform(numeric_features)

# # Create a DataFrame with the imputed values
# df_imputed = pd.DataFrame(numeric_features_imputed, columns=numeric_features.columns)

# # Reattach the URL column
# df_imputed['song_url'] = df['song_url']

# # Standardize the data
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(df_imputed[['valence', 'energy', 'tempo', 'danceability', 'harmonic_complexity', 'rhythmic_variability']])

# # Dimensionality reduction 
# pca = PCA(n_components=3)
# pca_features = pca.fit_transform(scaled_features)

# # Apply K-Means clustering
# kmeans = KMeans(n_clusters=4, random_state=42)
# clusters = kmeans.fit_predict(pca_features)

# # Add cluster labels to DataFrame
# df_imputed['cluster'] = clusters

# # Analyze clusters
# for cluster in range(4):
#     print(f"Cluster {cluster}")
#     print(df_imputed[df_imputed['cluster'] == cluster].describe())
    
#     print(f"Cluster {cluster} URLs:")
#     urls = df_imputed[df_imputed['cluster'] == cluster]['song_url'].tolist()
#     for url in urls:
#         print(url)
#     print() 
