import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

data_path = 'C:/Users/raian/Documents/git workspace/relatorios_machine_learning/MyDatasets/regressao/linear_noise/data.csv'
dados=pd.read_csv(data_path) #o meu arquivo com os dados está na mesma pasta que o arquivo do código

X = dados.iloc[:,0].values
Y = dados.iloc[:,1].values
plt.scatter(X, Y)
plt.show()

r = pearsonr(X, Y)
print(f'Coeficiente de correlação: {r}')

#Separando dados de treino e de teste
#utilizamos 70% dos dados para treino e o restante (30%) para teste.
x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.3)

#Precisamos redimensionar os dados para fazer a regressão linear
x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)
y_test=y_test.reshape(-1,1)

#treinando o modelo
reg = LinearRegression()
reg.fit(x_train,y_train)
pred = reg.predict(x_test)

plt.scatter(X, Y, color="blue")
plt.plot(x_test, pred, color="red")
plt.title("Regressão Linear")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

r_squared = r2_score(y_test, pred)
print(f'Coeficiente r2: {r_squared}') # Coeficiente r2: 0.9921214562343487

residual = y_test - pred

# plt.title('Resíduos')
# plt.xlabel('Resíduos (Dólar)')
# plt.ylabel('Frequência Absoluta')
# plt.hist(residual, rwidth=0.9)
# plt.show()