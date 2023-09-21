import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster
import pyclustering
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

# Task 0
# Load csv data
iris_data = pd.read_csv('A3/iris_clusters.csv', sep=";")

print(iris_data.head())

# Task 1
# k-Means clustering
k = 3
kmeans = sklearn.cluster.KMeans(n_clusters=k, n_init="auto")
kmeans.fit(iris_data)

# Plot 
# plt.scatter(iris_data["sw"], iris_data["sl"], c=kmeans.labels_, cmap='viridis')
# plt.show()

# number of records in each cluster
print("Number of records in each cluster:")
print(np.count_nonzero(kmeans.labels_ == 0))
print(np.count_nonzero(kmeans.labels_ == 1))
print(np.count_nonzero(kmeans.labels_ == 2))

# Task 2
# Preprocessing

# Remove outliers:
clf = LocalOutlierFactor(n_neighbors=20, contamination="auto")
outliers = clf.fit_predict(iris_data)
iris_data = iris_data[outliers == 1]

# Remove remaining outliers by hand (same as in A1)
iris_data = iris_data[iris_data.sw < 10]
iris_data = iris_data[iris_data.sl < 15]
iris_data = iris_data[iris_data.pl < 45]

# Normalize data using min-max normalization
scaler = sklearn.preprocessing.MinMaxScaler()
iris_data_scaled = scaler.fit_transform(iris_data)

# Cluster using k-means
kmeans = sklearn.cluster.KMeans(n_clusters=k, n_init="auto")
kmeans.fit(iris_data_scaled)

# Add cluster labels to dataframe
iris_data['cluster'] = list(kmeans.labels_)

# Number of records in each cluster
print("Number of records in each cluster after normalizing and removing outliers:")
print(np.count_nonzero(kmeans.labels_ == 0))
print(np.count_nonzero(kmeans.labels_ == 1))
print(np.count_nonzero(kmeans.labels_ == 2))

# Coords of centroids
print("Coordinates of centroids:")

for i in range(k):
    # print in order pw, pl, sw, sl
    print("pw: ", kmeans.cluster_centers_[i][1], end=" ")
    print("pl: ", kmeans.cluster_centers_[i][0], end=" ")
    print("sw: ", kmeans.cluster_centers_[i][3], end=" ")
    print("sl: ", kmeans.cluster_centers_[i][2], end=" ")
    print()


# Plot using seaborn
# sns.pairplot(iris_data, hue="cluster", palette="bright")
# plt.show()





