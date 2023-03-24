import matplotlib.pyplot as plt
import pandas as pd
import pickle

cities = pd.read_csv('cities.csv')
kmeans = pickle.load(open( "model.pkl", 'rb'))

k = 5

labels = kmeans.labels_;
# Plot the clustered data
scatter = plt.scatter(cities['Longitude'], cities['Latitude'], c=labels, cmap='jet')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(*scatter.legend_elements(),loc = 'best')
plt.title(f'{k}-means clustering of cities')
plt.savefig('./chart.png')
plt.show()
