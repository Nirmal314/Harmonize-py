import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

with open('json/songs_with_categories.json', 'r') as file:
    small_data = json.load(file)

selected_features = ["danceability", "energy", "acousticness", "valence"]
# selected_features = ["danceability", "energy", "loudness", "acousticness", "valence", "tempo"]

small_features = []
small_labels = []
for item in small_data:
    selected_item_features = [item['features'][key] for key in selected_features]
    small_features.append(selected_item_features)
    small_labels.append(item['category'])

df_small_features = pd.DataFrame(small_features, columns=selected_features)
df_small_labels = pd.Series(small_labels)

scaler = MinMaxScaler()
X_small_normalized = pd.DataFrame(scaler.fit_transform(df_small_features), columns=selected_features)

X_train, X_test, y_train, y_test = train_test_split(X_small_normalized, df_small_labels, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

df_large_unlabeled = pd.read_csv('1.2mln.csv')

X_unlabeled = df_large_unlabeled[selected_features]
X_unlabeled_normalized = pd.DataFrame(scaler.transform(X_unlabeled), columns=selected_features)

df_large_unlabeled['predicted_category'] = clf.predict(X_unlabeled_normalized)

df_large_unlabeled.to_csv('1.2mln_labeled.csv', index=False)