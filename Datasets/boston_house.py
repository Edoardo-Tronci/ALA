from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import openml
import datetime
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# print('******* BOSTON HOUSE PRICES ***********')

housesal = fetch_openml(name='house_sales',version=1,as_frame=True)

# print(housesal.data.shape)
# print(housesal.details['version'])

daystart = datetime.datetime.strptime('20140101','%Y%m%d').toordinal()
daystr = housesal['data']['date'][0].split('T')[0]
daysince = datetime.datetime.strptime(daystr, '%Y%m%d').toordinal() - daystart
# print(daystr,' ',daysince)
daystr = housesal['data']['date'][1].split('T')[0]
daysince = datetime.datetime.strptime(daystr, '%Y%m%d').toordinal() - daystart
# print(daystr,' ',daysince)

cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
           'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
           'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
           'sqft_living15', 'sqft_lot15']

(m,n) = housesal.data.shape

X = np.zeros((m,n-1))
X[:,1:n-1] = housesal.data[cols].to_numpy()
for i in range(m):
    daystr = housesal['data']['date'][i].split('T')[0]
    daysince = datetime.datetime.strptime(daystr, '%Y%m%d').toordinal() - daystart
    X[i,0] = daysince

y = housesal.data['price'].to_numpy()

# Xtr, Xts, ytr, yts = train_test_split(X,y,test_size=0.25,random_state=133)
# print(Xtr.shape," ",ytr.shape)
# print(Xts.shape," ",yts.shape)
# print(X.shape," ",y.shape)
# print('number of distinct values in target: ',len(np.unique(y)),'\n')

scalerx = StandardScaler()
scalery = StandardScaler()
scalerx.fit(X)
scalery.fit(y.reshape(-1,1))

X_train = scalerx.transform(X)
y_train = scalery.transform(y.reshape(-1,1)) 

# print(X_train.shape," ",y_train.shape)


