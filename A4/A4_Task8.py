import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import nltk
from nltk.corpus import stopwords


# Load dataset
dataset = load_files('A4/bbc',encoding='latin-1')
X_original, y = dataset.data, dataset.target

# Extract features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_original)

# kNN
n_neighbors = 1
model = KNeighborsClassifier(n_neighbors, metric="euclidean")
scores = cross_val_score(model, X, y, cv=10)
print("kNN with k=1, 10 fold cross validation: ", scores.mean(), " +/- ", scores.std())

# # Try different k
# k_values = [1, 3, 5, 7, 9, 11, 13, 15]
# for k in k_values:
#     model = KNeighborsClassifier(k, metric="euclidean")
#     scores = cross_val_score(model, X, y, cv=10)
#     print("kNN with k=", k, ", 10 fold cross validation: ", scores.mean(), " +/- ", scores.std())

# Preprocess data more

# Remove stop words (made no differnce)
# stop_words = set(stopwords.words('english'))
# vectorizer = CountVectorizer(stop_words=stop_words)
# vectorizer = CountVectorizer()

# X = vectorizer.fit_transform(X_original)

# Make lower case, x_orginal is a list of strings (made no difference)
# X = [x.lower() for x in X_original]
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(X)

# Remove unfrequent words
vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(2, 2), min_df=0.01, max_df=0.99)
X = vectorizer.fit_transform(X_original)



# Evaluate kNN
n_neighbors = 1
model = KNeighborsClassifier(n_neighbors, metric="euclidean", weights="uniform")
scores = cross_val_score(model, X, y, cv=10)
print("kNN with k=1, 10 fold cross validation, stop words removed: ", scores.mean(), " +/- ", scores.std())

# Loop over different parameters of count vectorizer
ngram_range_values = [(1, 1), (1, 2), (2, 2)]
min_df_values = [0.01, 0.05, 0.1]
max_df_values = [0.9, 0.95, 0.99]

max_score = 0

for ngram_range in ngram_range_values:
    for min_df in min_df_values:
        for max_df in max_df_values:
            vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=min_df, max_df=max_df)
            X = vectorizer.fit_transform(X_original)
            model = KNeighborsClassifier(n_neighbors, metric="euclidean", weights="uniform")
            scores = cross_val_score(model, X, y, cv=10)
            print("Results, ngram_range=", ngram_range, ", min_df=", min_df, ", max_df=", max_df, ": ", scores.mean(), " +/- ", scores.std())
            if scores.mean() > max_score:
                max_score = scores.mean()
                max_ngram_range = ngram_range
                max_min_df = min_df
                max_max_df = max_df

print("Best results, ngram_range=", max_ngram_range, ", min_df=", max_min_df, ", max_df=", max_max_df, ": ", max_score)


