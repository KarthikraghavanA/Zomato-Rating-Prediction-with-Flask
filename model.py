import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

zData = pd.read_csv('Prepocessed_Zomato_data.csv')
y = zData['rate']
X = zData.drop('rate', axis=1)

xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=.3,random_state=0)

# Preparing Extra Trees Regression

etr = ExtraTreesRegressor(n_estimators=120)
etr.fit(xtrain, ytrain)

ypred = etr.predict(xtest)

print(f'Training Score',etr.score(xtrain, ytrain))
print(f'Testing Score',etr.score(xtest, ytest))

import pickle
pickle.dump(etr, open('etrmodel.pkl','wb'))
model = pickle.load(open('etrmodel.pkl','rb'))

print(ypred)