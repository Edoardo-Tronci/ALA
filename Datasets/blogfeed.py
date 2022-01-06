from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.utils import shuffle

################ BLOGDATA
# print('************* BLOGDATA ****************')

DATA = np.genfromtxt('blogfeed.csv',delimiter=',')
Xtr = DATA[:52397,:-1]
ytr = DATA[:52397,-1]

# print(Xtr.shape," ",ytr.shape)
# print('number of distinct values in target: ',len(np.unique(ytr)),'\n')

scalerx = StandardScaler()
scalery = StandardScaler()
scalerx.fit(Xtr)
scalery.fit(ytr.reshape(-1,1))

X_train = scalerx.transform(Xtr)
y_train = scalery.transform(ytr.reshape(-1,1))
