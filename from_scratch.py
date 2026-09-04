import pandas as pd
import numpy as np

df = pd.read_csv('winequality-red.csv', sep=';')

print(df.columns)
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())

X = df.drop(columns='quality')
y = df['quality']

print(X.shape)
print(y.shape)

np.random.seed(42)
indices = np.random.permutation(len(X))
test_size = int(len(X)*0.2)

train_indices = indices[test_size:]
test_indices = indices[:test_size]

print(train_indices.shape)
print(test_indices.shape)

x_train = X.iloc[train_indices]
x_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]







