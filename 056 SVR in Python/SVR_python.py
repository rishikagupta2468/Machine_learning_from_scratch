#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1:].values

#preprocessing
from sklearn.preprocessing import StandardScaler
x_sc = StandardScaler()
X = x_sc.fit_transform(X)
y_sc = StandardScaler()
y = y_sc.fit_transform(y)

#SVM model creation
from sklearn.svm import SVR
regressor = SVR(kernel='rbf',degree=4)
regressor.fit(X,y)

#prediction
y_pred = regressor.predict(X)

#new prediction
ny = y_sc.inverse_transform(regressor.predict(x_sc.transform(np.array([[6.6]]))))
print(ny)

#visualization
plt.scatter(X,y,color='blue')
plt.plot(X,y_pred,color='red')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()