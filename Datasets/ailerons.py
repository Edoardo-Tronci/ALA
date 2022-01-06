from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime
import numpy as np
from sklearn.utils import shuffle

################ AILERONS
# print('************* AILERONS ****************')
ailerons = fetch_openml(name='ailerons',version=1)

# print(ailerons.data.shape)
# print(ailerons.details['version'])
X = ailerons.data
y = ailerons.target

# print(X.shape," ",y.shape)
# print('number of distinct values in target: ',len(np.unique(y)),'\n')

scalerx = StandardScaler()
scalery = StandardScaler()
scalerx.fit(X)
scalery.fit(y.values.reshape(-1,1))

X_train = scalerx.transform(X)
y_train = scalery.transform(y.values.reshape(-1,1))

# print(X_train.shape," ",y_train.shape) 
