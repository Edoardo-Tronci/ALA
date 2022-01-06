from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#import openml
import datetime
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# print('*************    Mv    ****************')

MV = fetch_openml(name='Mv',version=1)

# print(MV.data.shape)
# print(MV.details['version'])
X = MV.data
X = pd.get_dummies(data=X, columns=['x3', 'x7', 'x8'])
y = MV.target

# print(X.shape," ",y.shape)
# print('number of distinct values in target: ',len(np.unique(y)),'\n')

scalerx = StandardScaler()
scalery = StandardScaler()
scalerx.fit(X)
scalery.fit(y.values.reshape(-1,1))

X_train = scalerx.transform(X)
y_train = scalery.transform(y.values.reshape(-1,1)) 