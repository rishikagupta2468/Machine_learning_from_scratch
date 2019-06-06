#multiple linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#encoding categorical data
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#avoiding dummy variable trap
X = X[:,1:]

#splitting dataset into train test set
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)

#fitting multiple regression model
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train,y_train)

#predicting the test set
y_pred = linreg.predict(X_test)

#build optimal model with backward elimination
import statsmodels.formula.api as sm
X = np.append(arr= np.ones((50,1)).astype(int),values=X, axis=1)
X_opt = X[:,[0,1,2,3,4,5]]
linreg_OLS = sm.OLS(endog=y,exog=X_opt).fit()
linreg_OLS.summary()
X_opt = X[:,[0,1,3,4,5]]
linreg_OLS = sm.OLS(endog=y,exog=X_opt).fit()
linreg_OLS.summary()
X_opt = X[:,[0,3,4,5]]
linreg_OLS = sm.OLS(endog=y,exog=X_opt).fit()
linreg_OLS.summary()
X_opt = X[:,[0,3,5]]
linreg_OLS = sm.OLS(endog=y,exog=X_opt).fit()
print(linreg_OLS.summary())

#new predictiion
x_train,x_test,Y_train,Y_test = train_test_split(X_opt,y,test_size=0.2,random_state=0)
linreg.fit(x_train,Y_train)

Y_pred  = linreg.predict(x_test)