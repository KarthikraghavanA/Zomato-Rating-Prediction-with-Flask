import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings('ignore')


df = pd.read_csv('ZomatoProcessed.csv')

df.drop('Unnamed: 0', axis=1, inplace=True)
print(df.head())
x = df.drop('rate', axis=1)
y = df.rate

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=10)

# PREPARING EXTRA TREES REGRESSOR

et = ExtraTreesRegressor(n_estimators=120)
et.fit(xtrain, ytrain)

ypred = et.predict(xtest)

# SAVING THE MODEL TO DISK
pickle.dump(et, open('Zomatoetmodel.pkl', 'wb'))

model = pickle.load(open('Zomatoetmodel.pkl', 'rb'))
print(ypred)
