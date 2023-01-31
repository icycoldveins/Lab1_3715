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

# house1=np.array([3,1,1180,5650,1])
# house2=np.array([3,2,1680,8080,1])
# print("house1: {}".format(house1))
# print("house2: {}".format(house2))
# print("inner product: {}".format(np.inner(house1,house2)))



A = X[:, :3]
B = X[:, 3:]
C=np.transpose(B)
print(C.shape)