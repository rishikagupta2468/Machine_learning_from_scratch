import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Salary_Data.csv')

X= dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#splitting dataset into train test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#fitting simple regression model
from sklearn.linear_model import LinearRegression
linearrg = LinearRegression()
linearrg.fit(X_train,y_train)
#predicting
y_pred = linearrg.predict(X_test)

#visualization
#training set
plt.scatter(X_train,y_train,color='blue')
plt.plot(X_train,linearrg.predict(X_train),color='red')
plt.title('Salary predict')
plt.show()

#test set
plt.scatter(X_test,y_test,color='yellow')
plt.plot(X_test,y_pred,color='black')
plt.xlabel('year_experience')
plt.ylabel('salary')
plt.show()