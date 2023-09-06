import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyfpgrowth

# TASK 1
# Load csv data
raw_data = pd.read_csv('A2/www.csv', sep="\t")

# Count number of rows and columns
print("Size of dataframe: ", raw_data.shape)

# TASK 2
# Generate frequent itemsets

# Remove id and query
data = raw_data.drop(['UserId', 'Query'], axis=1)
data_list = data.values.tolist()

# Convert the DataFrame to a list of lists with integers indicating the column index
list_of_lists = []

for row in raw_data.values:
    row_as_integers = [i-2 for i, value in enumerate(row) if value == "t"]
    list_of_lists.append(row_as_integers)

# Find frequent patterns
patterns = pyfpgrowth.find_frequent_patterns(list_of_lists, 100)

# Print Patterns and their support counts
for value in patterns:
    if len(value) == 1:
        print(data.columns[value[0]], patterns[value])
    else:
        print(value, patterns[value])

print("Length of patterns: ", len(patterns))

# Support count for of is 958
print("Support count for of is: ", 958)
print("Support (fraction) for of is: ", 958 / 9999)
print("Support for support count = 100 is:", 100 / 9999) # 0.01






