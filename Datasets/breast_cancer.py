import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

################ BREAST CANCER
# print('*************   BREAST CANCER   ****************')

breast_cancer = load_breast_cancer()
# print(breast_cancer)

Xtr = breast_cancer.data
ytr = breast_cancer.target

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
