import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the Titanic dataset
df = pd.read_csv('train.csv')

# Data preprocessing (selecting relevant features and handling missing values)
# For this example, we'll focus on the 'Age' and 'Fare' features
selected_features = ['Age', 'Fare']
df = df[selected_features].dropna()

# Standardize the data (important for K-means algorithm)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Perform K-means clustering
num_clusters = 4  # You can choose the number of clusters based on your preference
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

# Data visualization
# Create a scatter plot for visualizing clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Fare', hue='cluster', palette='viridis', s=80, alpha=0.8)
plt.title('Clustering of Titanic Data')
plt.xlabel('Age')
plt.ylabel('Fare')

# Display the plot
plt.show()
