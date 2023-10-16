# importar pacotes necessários
import numpy as np
import matplotlib.pyplot as plt
# exemplo de plots determinísticos
np.random.seed(42)
det_x = np.arange(0,10,0.1)
det_y = 2 * det_x + 3
# exemplo de plots não determinísticos
non_det_x = np.arange(0, 10, 0.1)
non_det_y = 2 * non_det_x + np.random.normal(size=100)
# plotar determinísticos vs. não determinísticos
fig, axs = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
axs[0].scatter(det_x, det_y, s=2)
axs[0].set_title("Determinístico")
axs[1].scatter(non_det_x, non_det_y, s=2)
axs[1].set_title("Não Determinístico")
plt.show()
# importar os pacotes necessários
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# criar modelo linear e otimizar
lm_model = LinearRegression()
lm_model.fit(non_det_x.reshape(-1,1), non_det_y)
# extrair coeficientes
slope = lm_model.coef_
intercept = lm_model.intercept_
# imprimir os valores encontrados para os parâmetros
print("b0: \t{}".format(intercept))
print("b1: \t{}".format(slope[0]))
# Será impresso os seguintes valores:
# b0: 	-0.17281285407737457
# b1: 	2.0139325932693497