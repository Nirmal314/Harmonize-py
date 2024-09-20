import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file with predicted labels
df_labeled = pd.read_csv('1.2mln_labeled.csv')

# Summary statistics for the features
print("\nSummary Statistics for Features:")
print(df_labeled[['danceability', 'energy', 'acousticness', 'valence']].describe())

# Count the number of songs in each predicted category
category_counts = df_labeled['predicted_category'].value_counts()
print("\nNumber of Songs in Each Category:")
print(category_counts)

# Plot the distribution of categories
plt.figure(figsize=(10, 6))
category_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Predicted Categories')
plt.ylabel('Number of Songs')
plt.xlabel('Category')
plt.show()

# Plot the distribution of each feature by predicted category
features = ['danceability', 'energy', 'acousticness', 'valence']

for feature in features:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='predicted_category', y=feature, data=df_labeled, palette='Set2')
    plt.title(f'Distribution of {feature} by Predicted Category')
    plt.ylabel(feature)
    plt.xlabel('Category')
    plt.xticks(rotation=45)
    plt.show()

# Correlation matrix to understand feature relationships
plt.figure(figsize=(10, 8))
correlation_matrix = df_labeled[features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()

# Pairplot to visualize relationships between features and categories
# sns.pairplot(df_labeled, hue='predicted_category', vars=features, palette='Set2')
# plt.suptitle('Pairplot of Features by Predicted Category', y=1.02)
# plt.show()
