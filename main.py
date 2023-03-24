import pandas as pd
from sklearn.cluster import KMeans
import pickle

# Load the dataset
cities = pd.read_csv('cities.csv')

# Choose the number of clusters
k = 5

# Initialize the k-means model
kmeans = KMeans(n_clusters=k)

# Fit the model to the data
kmeans.fit(cities[['Latitude', 'Longitude']])

# Get the cluster assignments for each data point
labels = kmeans.labels_

# Print the cluster assignments
print(labels)

#print(dict(zip(cities.City.values, labels)))
pickle.dump(kmeans,open( "model.pkl", 'wb'))

