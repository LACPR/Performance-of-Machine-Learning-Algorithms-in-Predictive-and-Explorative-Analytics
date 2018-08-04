# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('diabetic_data.csv')

transactions = []
for i in range(0, 101766):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 50)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.002, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
