import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# print('************* APPLIANCES ENERGY PREDICTION ****************')
df = pd.read_csv('appliances.csv', sep=',',header=0)

# print(df.head())
# print(df.values[0,0])

m, nn = df.shape

ytr = df.values[:,1:3]

Xtr = np.hstack((np.reshape(range(m),(m,1)),df.values[:,3:-2]))

# print(Xtr.shape," ",ytr.shape)
# print('number of distinct values in target: ',len(np.unique(ytr)),'\n')

scalerx = StandardScaler()
scalery = StandardScaler()
scalerx.fit(Xtr)
scalery.fit(ytr)

X_train = scalerx.transform(Xtr)
y_train = scalery.transform(ytr)