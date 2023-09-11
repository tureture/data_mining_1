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
patterns = pyfpgrowth.find_frequent_patterns(list_of_lists, 39)

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
print()

# TASK 3
# Find frequent patterns for different support counts
for i in np.linspace(0.001, 0.01, 4):
    patterns_2 = pyfpgrowth.find_frequent_patterns(list_of_lists, round(i * 9999))
    print("Nr itemsets above threshold for", i, "is", len(patterns_2))
    print("Current threshold is", int(i*9999))


# TASK 4
# Generate association rules
rules = pyfpgrowth.generate_association_rules(patterns, 0.8)

# Print rules and their confidence
print()
print("Length of rules: ", len(rules))
for rule in rules:
    for index in rule:
        print(data.columns[index], end=" ")
    temp = rules[rule]
    print("--> ", end="")
    for index in temp[0]:
        print(data.columns[index], end=" ")
    
    print(" Confidence: ", temp[1])

print()

# TASK 5
# Find association rules for different confidence values
for i in np.linspace(0.1, 1, 10):
    rules_2 = pyfpgrowth.generate_association_rules(patterns, i)
    print("Nr rules for confidence", i, "is", len(rules_2))


# TASK 6
# Find some interesting rules
rules_3 = pyfpgrowth.generate_association_rules(patterns, 0.01)

# Print rules and their confidence
print()
print("Length of rules: ", len(rules))
for i, rule in enumerate(rules_3):
    for index in rule:
        print(data.columns[index], end=" ")
    temp = rules_3[rule]
    print("--> ", end="")
    for index in temp[0]:
        print(data.columns[index], end=" ") 

    print(" Confidence: ", temp[1], end=" ")
    print("Rule nr: ", i)

# Select 3 interesting rules, sort by lift and confidence
# real -> estate (2) Lift =130, C = 0.785
# school -> high (10) L = 38.99, C = 0.393
# what -> is (12) L = 34.26, C = 0.390

# Select 3 unexpected rules
# in -> of (19) L = 0.688, C = 0.0659
# of -> in (4) L = 0.688, C = 0.057
# to how -> a (17) L = 10.1, C = 0.0322

