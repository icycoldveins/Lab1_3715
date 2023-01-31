import numpy as np
import pandas as pd

# load data with pandas
df = pd.read_csv('cars.csv')
# print(df)

# convert a pandas dataframe to a numpy array
X = df.values[:, 1:-1]
# print(type(X))
# print(X.shape)
# print(X)

# assign variables to first and second to last column
X1 = X[:, 0]
X2 = X[:, -2]


# 1. addition and subtraction
result_add = X1 + X2
result_sub = X1 - X2
print(result_sub)
# 2. element-wise multiplication
result_mul = np.multiply(X1, X2)
# 3. inner product

# 4. three norms

# 5. compute the distance between these two vectors


