from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime
import numpy as np
from sklearn.utils import shuffle

# print('************* ARCENE ****************')
arcene = fetch_openml(name='arcene',version=1)

# print(arcene.data.shape)
# print(arcene.details['version'])
X = arcene.data
y = arcene.target
y = y.to_frame()

# print(X.shape," ",y.shape)
# print('number of distinct values in target: ',len(np.unique(y)),'\n')

scalerx = StandardScaler()
scalery = StandardScaler()
scalerx.fit(X)
scalery.fit(y.values.reshape(-1,1))

X_train = scalerx.transform(X)
y_train = scalery.transform(y.values.reshape(-1,1)) 