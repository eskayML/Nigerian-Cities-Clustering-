import matplotlib.pyplot as plt
import pandas as pd
import pickle

cities = pd.read_csv('cities.csv')
kmeans = pickle.load(open( "model.pkl", 'rb'))

k = 5

labels = kmeans.labels_;
# Plot the clustered data
plt.scatter(cities['Latitude'], cities['Longitude'], c=labels, cmap='jet')
plt.ylabel('Longitude')
plt.xlabel('Latitude')
plt.legend(loc = 'best')
plt.title(f'{k}-means clustering of cities')
plt.savefig('./chart.png')
plt.show()
