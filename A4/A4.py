import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

# Task 0 
# Load csv data
cancer_data = pd.read_csv('A4/cancer.csv', sep=";")

# Number of columns and rows
print("Number of columns: ", len(cancer_data.columns))
print("Number of rows: ", len(cancer_data))

# Count number of B and M
print("Number of B: ", np.count_nonzero(cancer_data.diagnosis == "B"))
print("Number of M: ", np.count_nonzero(cancer_data.diagnosis == "M"))



