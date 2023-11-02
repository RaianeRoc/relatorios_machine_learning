import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR 
from sklearn import metrics

#Importando os dados
data_path = 'C:/Users/raian/Documents/git workspace/relatorios_machine_learning/MyDatasets/regressao/sawtooth/data.csv'
dataset=pd.read_csv(data_path) #o meu arquivo com os dados está na mesma pasta que o arquivo do código
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

y_p = y.reshape(-1,1)
from sklearn.preprocessing import StandardScaler
StdS_X = StandardScaler()
StdS_y = StandardScaler()
X_l = StdS_X.fit_transform(X)
y_p = StdS_y.fit_transform(y_p)

# plt.scatter(X_l, y_p, color = 'red') # plotting the training set
# plt.title('Scatter Plot') # adding a tittle to our plot
# plt.xlabel('Levels') # adds a label to the x-axis
# plt.ylabel('Salary') # adds a label to the y-axis
# plt.show() # prints

# create the model object
regressor = SVR(kernel = 'poly')
# fit the model on the data
regressor.fit(X_l, y_p)
pred = StdS_y.inverse_transform(regressor.predict(X_l).reshape(-1,1))
print('R2 Value:',metrics.r2_score(y_p, pred)) # RBF -> R2 Value: 0.8712587552057448
                                                # linear -> R2 Value: 0.6722366728546595
                                                # poly -> R2 Value: 0.3732474053644833
# # inverse the transformation to go back to the initial scale
plt.scatter(StdS_X.inverse_transform(X_l), StdS_y.inverse_transform(y_p), color = 'red')
plt.plot(StdS_X.inverse_transform(X_l), StdS_y.inverse_transform(regressor.predict(X_l).reshape(-1,1)), color = 'blue')
# add the title to the plot
plt.title('Support Vector Regression Model')
# label x axis
plt.xlabel('Position')
# label y axis
plt.ylabel('Salary Level')
# print the plot
plt.show()

plt.title('Regressão SVR')

# plt.show()