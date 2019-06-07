#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#SVM model creation
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#prediction
y_pred = regressor.predict(X)

#new prediction
ny = regressor.predict(np.array([[6.6]]))

#visualization
plt.scatter(X,y,color='blue')
plt.plot(X,y_pred,color='red')
plt.xlabel('position level')
plt.ylabel('salary')
plt.title('decision tree regression')
plt.show()

#visualization with high resolution
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='blue')
plt.plot(X_grid,regressor.predict(X_grid),color='red')
plt.xlabel('position level')
plt.ylabel('salary')
plt.title('decision tree regression')
plt.show()