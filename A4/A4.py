import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn import preprocessing

# Task 0 
# Load csv data
cancer_data = pd.read_csv('A4/cancer.csv', sep=";")

# Number of columns and rows
print("Number of columns: ", len(cancer_data.columns))
print("Number of rows: ", len(cancer_data))

# Count number of B and M
print("Number of B: ", np.count_nonzero(cancer_data.diagnosis == "B"))
print("Number of M: ", np.count_nonzero(cancer_data.diagnosis == "M"))


# Task 1
X, y = cancer_data.iloc[:, 2:], cancer_data.iloc[:, 1]


# k-NN classifier
n_neighbors = 1
model = KNeighborsClassifier(n_neighbors, metric="euclidean")

# Holdout 70/30
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X,y, test_size=0.3, random_state=0)
model.fit(X_trainset, y_trainset)
model.score(X=X_testset,y=y_testset)
print("Holdout 70/30: ", model.score(X=X_testset,y=y_testset))


# 10 fold cross validation
scores = cross_val_score(model, X, y, cv=10)
print("10 fold cross validation: ", scores.mean(), " +/- ", scores.std())

# Leave one out
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
print("Leave one out: ", scores.mean(), " +/- ", scores.std())

# 10 fold cross validation with k=10
n_neighbors = 10
model = KNeighborsClassifier(n_neighbors, metric="euclidean")
scores = cross_val_score(model, X, y, cv=10)
print("10 fold cross validation with k=10: ", scores.mean(), " +/- ", scores.std())

# Task 2
n_neighbors = 1
model = KNeighborsClassifier(n_neighbors, metric="euclidean")

# Scale data
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Leave one out cross validation
loo = LeaveOneOut()
scores = cross_val_score(model, X_scaled, y, cv=loo)
print("Leave one out cross validation (after scaling): ", scores.mean(), " +/- ", scores.std())



