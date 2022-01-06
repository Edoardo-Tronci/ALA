import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import fetch_openml

# print('************* MNIST HANDWRITTEN DIGIT ****************')

# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = X / 255.0

# print("X:", X.shape, "y:", y.shape)
# print('number of distinct values in target: ',len(np.unique(y)),'\n')

X_train = X.to_numpy()

y_train = y.to_numpy()
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y_train.reshape(-1,1))
y_train = enc.transform(y_train.reshape(-1,1)).toarray()

# y_train = y.to_numpy().astype('uint8')

# print("X_train:", X_train.shape, "y_train:", y_train.shape)




