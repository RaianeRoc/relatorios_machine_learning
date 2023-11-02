import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
pd.set_option('display.max_columns', 500)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import kurtosis
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_breuschpagan, het_goldfeldquandt,het_white
from statsmodels.stats.diagnostic import linear_harvey_collier, linear_reset, spec_white
from statsmodels.stats.diagnostic import linear_rainbow
from statsmodels.graphics.regressionplots import plot_leverage_resid2
from yellowbrick.regressor import CooksDistance
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from sklearn.linear_model import LinearRegression

#Importando os dados
data_path = 'C:/Users/raian/Documents/git workspace/relatorios_machine_learning/MyDatasets/regressao/abalone/data.csv'
df=pd.read_csv(data_path) #o meu arquivo com os dados está na mesma pasta que o arquivo do código

#Visualizando tabela dos dados
X = df.iloc[:,0:-2]
y = df.iloc[:,-1]
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# Criando o modelo LinearRegression
regr = LinearRegression()
# Realizar treinamento do modelo
regr.fit(X_train, y_train)
# Realizar predição com os dados separados para teste
y_pred = regr.predict(X_test)
# Visualização dos 20 primeiros resultados
plt.scatter(X, y, color="blue")
plt.plot(X_train, y_pred, color="red")
plt.title("Índice de Massa Corporal vs Custo do Seguro (Dados de Teste)")
plt.xlabel("Índice de Massa Corporal da Cliente")
plt.ylabel("Custo do Seguro (Dólares)")
plt.show()

#Breve descrição dos dados
# print(df.describe())
#Tabela de correlação
# corr = df.info()
# print(corr)