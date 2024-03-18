import pandas as pd
import numpy as np
import csv
from utils import *


# Reading data:
R = FoodData_preprocessing()

# matrix factorization:
k = 10  # len of embeddings
User_Embedding, PairItems_Embedding = mf(R, k, n_epoch=100, lr=.003, l2=.04) # Matrix Factorization


# Clustering with Shapley value:
clusters = cluster_users(User_Embedding, threshold=-1.0, max_users=5)
print("Clusters:", clusters)
