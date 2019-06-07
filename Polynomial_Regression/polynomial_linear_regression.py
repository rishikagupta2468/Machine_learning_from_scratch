#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#linear regression model
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X,y)
y_pred_lin = linreg.predict(X)

#build polynomial dataset
# y = bx^2
x=X**2
x = np.append(arr=X,values=x,axis=1)
x = x[:,1:]
polreg = LinearRegression()
polreg.fit(x,y)
y_pred_pol = polreg.predict(x)

#y= b0 + b1x + b2x^2
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
polyreg = LinearRegression()
polyreg.fit(X_poly,y)
y_pred_poly = polyreg.predict(X_poly)

#comparing all results
plt.scatter(X,y,color='black')
plt.plot(X,y_pred_lin,color='red')
plt.plot(x,y_pred_pol,color='yellow')
plt.plot(X,y_pred_poly,color='blue')
plt.show()