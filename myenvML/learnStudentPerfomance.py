import pandas as pd
import numpy as np


studentsPerfomance = pd.read_csv('./StudentsPerformance.csv')
# print(studentsPerfomance.head())
# print(studentsPerfomance.head(10))
# print(studentsPerfomance.tail())
# print(studentsPerfomance.iloc[0:5,0:3])
print(studentsPerfomance.iloc[[1,2,3,5],:])



