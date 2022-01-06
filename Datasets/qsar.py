import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# print('************* QSAR  ****************')
df = pd.read_csv('qsar.csv', delimiter = ";", header = None)
# print(df.shape)

last_ix = len(df.columns) - 1
Xtr, ytr = df.drop(last_ix, axis=1).to_numpy(), df[last_ix].to_numpy()

ytr = LabelEncoder().fit_transform(ytr).reshape(-1, 1)

# print(Xtr.shape," ",ytr.shape)
# print('number of distinct values in target: ',len(np.unique(ytr)),'\n') 

scalerx = StandardScaler()
scalery = StandardScaler()
scalerx.fit(Xtr)
scalery.fit(ytr.reshape(-1,1))

X_train = scalerx.transform(Xtr)
y_train = scalery.transform(ytr.reshape(-1,1))