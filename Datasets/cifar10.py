import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

# print('*************  CIFAR_10 ****************')

Xtr, ytr = fetch_openml('CIFAR_10', version=1, return_X_y=True)

ytr = ytr.to_frame()

# print(Xtr.shape," ",ytr.shape)
# print('number of distinct values in target: ',len(np.unique(ytr)),'\n')

scalerx = StandardScaler()
scalery = StandardScaler()
scalerx.fit(Xtr)
scalery.fit(ytr.values.reshape(-1,1))

X_train = scalerx.transform(Xtr)
y_train = scalery.transform(ytr.values.reshape(-1,1)) 

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y_train.reshape(-1,1))
y_train = enc.transform(y_train.reshape(-1,1)).toarray()