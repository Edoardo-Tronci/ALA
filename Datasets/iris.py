import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

################ IRIS
# print('*************   IRIS   ****************')

iris = load_iris()
# print(iris)

Xtr = iris.data
ytr = iris.target

# print(Xtr.shape," ",ytr.shape)
# print('number of distinct values in target: ',len(np.unique(ytr)),'\n')

scalerx = StandardScaler()
scalery = StandardScaler()
scalerx.fit(Xtr)
scalery.fit(ytr.reshape(-1,1))

X_train = scalerx.transform(Xtr)
y_train = scalery.transform(ytr.reshape(-1,1)) 

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y_train)
y_train = enc.transform(y_train).toarray()