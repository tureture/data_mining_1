import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import csv

# TASK 1

# 1.1 
# Read iris_data.csv and iris_labels.csv into pandas dataframes

iris_data = pd.read_csv('A1/iris_data.csv', sep=";")
iris_labels = pd.read_csv('A1/iris_labels.csv', sep=";")


# 1.2 
# Merge dataframes
data = pd.merge(iris_data, iris_labels, on='id', how="inner")

# 1.3 
# Filter out examiner column
data.drop(['examiner'], axis=1, inplace=True)

# 1.4
# Sort by species 
data.sort_values(by=['species'], inplace=True)

# 1.5
# Plot using seaborn
# sns.pairplot(data, hue="species")
# plt.show()

# Questions part 1
# What are the average length of sepals (sl) and their standard deviation?
print()
print("Average length of sepals(sl): ", data["sl"].mean())
print("Standard deviation of sepals(sl): ", data["sl"].std())
print("Number of instances for each class: ", data.groupby('species').count())
print()

# TASK 2

# 2.1
# Remove -9999 values from dataset
data = data[data.sl != -9999]
print("Average sepal length after missing values removed: ", data["sl"].mean())
print("Std sepal length after missing values removed: ", data["sl"].std())

# 2.2
# Plot using seaborn
#sns.pairplot(data, hue="species")
#plt.show()


# Remove remaining outliers
data = data[data.sw < 10]
data = data[data.sl < 20]

# Plot using seaborn
#sns.pairplot(data, hue="species")
#plt.show()

# Questions part 2
print("Average sepal length after outliers removed: ", data["sl"].mean())
print("Std sepal length after outliers removed: ", data["sl"].std())



# TASK 3

# 3.1
# Normalize data with MinMax 
#data.drop(['species'], axis=1, inplace=True)
# scaled = MinMaxScaler().fit_transform(data[["sl", "sw", "pl", "pw"]])
data_minmax = data.copy()
for col in data_minmax.columns:
    if col != 'species':
        data_minmax[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

print()
print("Mean of sl after MinMax: ", data_minmax["sl"].mean())
print("Std of sl after MinMax: ", data_minmax["sl"].std())

# 3.2
# Normalize data with mean and std
data_meanstd = data.copy()
for col in data_meanstd.columns:
    if col != 'species':
        data_meanstd[col] = (data[col] - data[col].mean()) / data[col].std()

print()
print("Mean of sl after mean and std: ", data_meanstd["sl"].mean())
print("Std of sl after mean and std: ", data_meanstd["sl"].std())

# 3.3 & 3.4
# Apply PCA
pca = PCA(n_components=0.95)
principalComponents = pca.fit(data_meanstd[["pl", "pw", "sl", "sw"]])
print("Nr components: ", principalComponents.n_components_)
print("Explained variance: ", sum(principalComponents.explained_variance_ratio_))
print("Values of components: ", principalComponents.components_)

# 3.5
data_meanstd_rescaled = data_meanstd.copy()

# rescale column in pl to range 0 - 100
data_meanstd_rescaled['pl'] = (data_meanstd_rescaled['pl'] - data_meanstd_rescaled['pl'].min()) / (data_meanstd_rescaled['pl'].max() - data_meanstd_rescaled['pl'].min())* 100

# Apply PCA
pca = PCA(n_components=0.95)
principalComponents = pca.fit(data_meanstd_rescaled[["pl", "pw", "sl", "sw"]])
print()
print("Rescaled Nr components: ", principalComponents.n_components_)
print("Rescaled Explained variance: ", sum(principalComponents.explained_variance_ratio_))
print("Rescaled Values of components: ", principalComponents.components_)




# TASK 4

# 4.1
# Sample 150 instances
sampled_data = data.sample(n=150, random_state=1)

# 4.2 
# Sample with bootstrap
sampled_data_boot = data.sample(n=150, replace=True, random_state=1)

# 4.3
# Stratified sampling
sampled_data_stratified = data.groupby('species', group_keys=False).apply(lambda x: x.sample(frac=0.5))

# 4.4
# Stratified sampling with 50 samples each
sampled_data_stratified_2 = data.groupby('species', group_keys=False).apply(lambda x: x.sample(50))

# Print nr of each type
print()
print("Nr of each type in sampled data: ", sampled_data.groupby('species').count())
print()
print("Nr of each type in sampled data with bootstrap: ", sampled_data_boot.groupby('species').count())
print()
print("Nr of each type in sampled data with stratified sampling: ", sampled_data_stratified.groupby('species').count())
print()
print("Nr of each type in sampled data with stratified sampling 50 each: ", sampled_data_stratified_2.groupby('species').count())