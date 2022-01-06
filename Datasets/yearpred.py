import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# print('************* YEARPRED  ****************')
df = pd.read_csv('yearpred.csv', sep=',',header=None)

ytr = df.values[:,0]
Xtr = df.values[:,1:]

# print(Xtr.shape," ",ytr.shape)
# print('number of distinct values in target: ',len(np.unique(ytr)),'\n')

scalerx = StandardScaler()
scalery = StandardScaler()
scalerx.fit(Xtr)
scalery.fit(ytr.reshape(-1,1))

X_train = scalerx.transform(Xtr)
y_train = scalery.transform(ytr.reshape(-1,1)) 

X_train = X_train[:100000,:]
y_train = y_train[:100000]

# print(X_train.shape," ",y_train.shape)