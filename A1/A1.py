import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
print(data.describe())
print("Average length of sepals(sl): ", data["sl"].mean())
print("Standard deviation of sepals(sl): ", data["sl"].std())
print("Number of instances for each class: ", data.groupby('species').count())



