import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

with open('json/songs_with_categories.json', 'r') as file:
    data = json.load(file)

selected_features = [
    "danceability",
    "energy", 
    # "loudness", 
    "acousticness",
    "valence", 
    # "tempo"
]

all_features = []
for item in data:
    selected_item_features = {key: item['features'][key] for key in selected_features}
    all_features.append(selected_item_features)
df_features = pd.DataFrame(all_features)

df_normalized = (df_features - df_features.min()) / (df_features.max() - df_features.min())

for i, item in enumerate(data):
    normalized_features = df_normalized.iloc[i].to_dict()
    item['normalized_features'] = {key: normalized_features[key] for key in selected_features}

categories = list(set(item['category'] for item in data))

category_features = {category: {feature: [] for feature in selected_features} for category in categories}

for item in data:
    category = item['category']
    for feature, value in item['normalized_features'].items():
        category_features[category][feature].append(value)

category_averages = {
    category: {feature: np.mean(values) for feature, values in features.items()} 
    for category, features in category_features.items()
}

df_category_averages = pd.DataFrame(category_averages).T

feature_variability = df_category_averages.std()

plt.figure(figsize=(10, 6))
feature_variability.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Feature Variability Across Categories (Standard Deviation)')
plt.ylabel('Standard Deviation')
plt.xlabel('Features')
plt.show()

most_varying_feature = feature_variability.idxmax()
least_varying_feature = feature_variability.idxmin()

print(f"Feature with the most variability: {most_varying_feature}")
print(f"Feature with the least variability: {least_varying_feature}")

df_normalized['category'] = [item['category'] for item in data]

num_features = len(selected_features)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
fig.suptitle('Feature Importance by Category', fontsize=16)

axes = axes.flatten()

for i, feature in enumerate(selected_features):
    ax = axes[i]
    df_normalized.boxplot(column=feature, by='category', grid=False, ax=ax)
    ax.set_title(feature)
    # ax.set_xlabel('Category')
    # ax.set_ylabel('Value')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
