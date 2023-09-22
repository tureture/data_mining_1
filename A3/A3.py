import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
import sklearn.cluster
import sklearn.metrics
import pyclustering
from sklearn.neighbors import LocalOutlierFactor
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

import numpy as np
import math

# Task 0
# Load csv data
iris_data_original = pd.read_csv('A3/iris_clusters.csv', sep=";")
iris_data = iris_data_original.copy()

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

# Task 3
# Choice of k and internal validation

# Check all k values from 2 to 10
max_db = 0
max_ki = 0
for i in range(2, 11):
    k = i
    data = iris_data_original.copy()

    # find clusters
    kmeans = sklearn.cluster.KMeans(n_clusters=k, n_init="auto")
    kmeans.fit(data)


    metric = sklearn.metrics.davies_bouldin_score(data, kmeans.labels_)
    if metric > max_db:
        max_db = metric
        max_ki = i
    print("Davies-Bouldin score for k =", k, "is", metric)

print("Best k is", max_ki, "with Davies-Bouldin score", max_db)

# Task 4
# Use single linkage clustering
iris_data = iris_data_original.copy()

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# Single linkage clustering
# model = AgglomerativeClustering(distance_threshold=None, n_clusters=2, linkage="single")
# model = AgglomerativeClustering(distance_threshold=None, n_clusters=2, linkage="complete")
model = AgglomerativeClustering(distance_threshold=None, n_clusters=2, linkage="average")
model = model.fit(iris_data)

# Plot dendrogram
# plt.title('Hierarchical Clustering Dendrogram')
# plot_dendrogram(model, truncate_mode='level', p=3)
# plt.show()

# Count number of records in highest 2 clusters manually
print()
print("Number of records in each cluster after single linkage clustering:")
print(len(iris_data))
print(np.count_nonzero(model.labels_ == 0))
print(np.count_nonzero(model.labels_ == 1))

# Task 5
# Clustering using dbscan
X = iris_data_scaled.copy()
# X = sklearn.preprocessing.StandardScaler().fit_transform(X)

db = sklearn.cluster.DBSCAN(eps=0.2, min_samples=5).fit(X)
labels = db.labels_

# eps = 1 gives 1 cluster??

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# Numbers of clusters
print()
print("Number of clusters after dbscan:")
print(n_clusters_)
print(n_noise_)

# Plot results from Dbscan

unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.show()


def k_distances(X, n):
    if type(X) is pd.DataFrame:
        X = X.values
    k = 0
    if n is None:
        k = X.shape[1] + 2
    else:
        k = n + 1
    dist_func = lambda x, y: math.sqrt(np.sum(np.power(x - y, np.repeat(2, x.size))))
    Distances = pd.DataFrame({
        "i": [i // 10 for i in range(0, len(X) * len(X))],
        "j": [i % 10 for i in range(0, len(X) * len(X))],
        "d": [dist_func(x, y) for x in X for y in X]
    })
    
    # Iterate through groups and calculate k-distances
    k_distances_list = []
    for group_name, group_data in Distances.groupby(by="i"):        
        # Check if k is within bounds for this group
        if k < len(group_data):
            k_distances_group = group_data["d"].iloc[k]
            k_distances_list.append(k_distances_group)
    
    return np.sort(k_distances_list)

# TODO: add your parameters here.
# data -- your normalized dataset (dataframe matrix)
# k    -- k-th neighbour (for distance metric). By default, k=count(features)+1
plt.figure()
d = k_distances(
    X,
    5
)
plt.plot(d)
plt.ylabel("k-distances")
plt.grid(True)
plt.show()

