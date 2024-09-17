import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import numpy as np
df = pd.read_csv('songs_uncleaned.csv')

columns_to_drop = ['Unnamed: 0', 'artists', 'album_name', 'popularity', 'duration_ms', 'explicit', 'time_signature', 'track_genre', 'instrumentalness', 'liveness', 'speechiness']

df = df.drop(columns=columns_to_drop)
df = df.dropna()
df = df.drop_duplicates()

columns_to_normalize = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']
columns_to_standardize = ['loudness']

min_max_scaler = MinMaxScaler()
df[columns_to_normalize] = min_max_scaler.fit_transform(df[columns_to_normalize])

# Standardization (Z-score Normalization)
standard_scaler = StandardScaler()
df[columns_to_standardize] = standard_scaler.fit_transform(df[columns_to_standardize])

Q1 = df['loudness'].quantile(0.25)
#? Impute outliers 

#! loudness: 5026 => 2333
#! tempo: 514 => 20
#! danceability: 474 => 97
#! energy: 0
#! acousticness: 0
#! valence: 0

Q3 = df['loudness'].quantile(0.75)
IQR = Q3-Q1
outliers_iqr = df[((df['loudness'] < (Q1-1.5*IQR)) | (df['loudness'] > (Q3+1.5*IQR)))]

median_to_impute = df[((df['loudness'] >= (Q1-1.5*IQR)) | (df['loudness'] <= (Q3+1.5*IQR)))]['loudness'].median()
df.loc[((df['loudness'] < (Q1-1.5*IQR)) | (df['loudness'] >(Q3+1.5*IQR))), 'loudness'] = median_to_impute

Q1 = df['tempo'].quantile(0.25)
Q3 = df['tempo'].quantile(0.75)
IQR = Q3-Q1
outliers_iqr = df[((df['tempo'] < (Q1-1.5*IQR)) | (df['tempo'] > (Q3+1.5*IQR)))]

median_to_impute = df[((df['tempo'] >= (Q1-1.5*IQR)) | (df['tempo'] <= (Q3+1.5*IQR)))]['tempo'].median()
df.loc[((df['tempo'] < (Q1-1.5*IQR)) | (df['tempo'] >(Q3+1.5*IQR))), 'tempo'] = median_to_impute

Q1 = df['danceability'].quantile(0.25)
Q3 = df['danceability'].quantile(0.75)
IQR = Q3-Q1
outliers_iqr = df[((df['danceability'] < (Q1-1.5*IQR)) | (df['danceability'] > (Q3+1.5*IQR)))]

median_to_impute = df[((df['danceability'] >= (Q1-1.5*IQR)) | (df['danceability'] <= (Q3+1.5*IQR)))]['danceability'].median()
df.loc[((df['danceability'] < (Q1-1.5*IQR)) | (df['danceability'] >(Q3+1.5*IQR))), 'danceability'] = median_to_impute


Q1 = df['loudness'].quantile(0.25)
Q3 = df['loudness'].quantile(0.75)
IQR = Q3-Q1
outliers_iqr = df[((df['loudness'] < (Q1-1.5*IQR)) | (df['loudness'] > (Q3+1.5*IQR)))]

# print(df.columns)       
# print(df.info())
# print(df.describe())

# print("Variance for normalized columns:")
# print(df[columns_to_normalize].var())

# print("Variance for standardized columns:")
# print(df[columns_to_standardize].var())

df.to_csv('songs_cleaned_withoutoutliers.csv', index=False)
with open('model/min_max_scaler.pkl', 'wb') as file:
    pickle.dump(min_max_scaler, file)

with open('model/standard_scaler.pkl', 'wb') as file:
    pickle.dump(standard_scaler, file)