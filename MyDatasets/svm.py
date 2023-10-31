import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('real/iris_bin_2/data.csv')

# Separando o dataset por classe
X = np.array(dataset.iloc[:, 1:-1])
y = np.array(dataset.iloc[:, -1])

# print(X.shape, y.shape)

# Separando dados de treino/teste
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, stratify=y, test_size=0.2)

# print(X_train_raw.shape, y_train_raw.shape)
# print(X_test_raw.shape, y_test_raw.shape)

# Adicionar uma coluna de 1s para considerar o termo de bias (intercept) no modelo linear
X_train = np.column_stack((X_train_raw, np.ones(X_train_raw.shape[0])))
X_test = np.column_stack((X_test_raw, np.ones(X_test_raw.shape[0])))

# Colocar y em formato de vetor (one hot)
def one_hot_convert(vec):
    matrix = []
    for idx in vec:
        m = np.zeros((3, 1))
        m[idx] = 1
        matrix.append(m)
    return np.array(matrix)

y_train = one_hot_convert(y_train_raw).reshape(y_train_raw.shape[0], -1)
y_test = y_test_raw.reshape(y_test_raw.shape[0], -1)

# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

# Funções de ativação para o neurônio
def activate_functions(type, matrix):
    if type == 'sigmoid':
        return 1 / (1 + np.exp(-matrix))
    elif type == 'softmax':
        exp_matrix = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
        return exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)   
    elif type == 'tanh':
        return np.tanh(matrix)
    elif type == 'step':
        return np.heaviside(matrix, 1)

# Função de treino para o classificador perceptron logístico
def train_logistic_perceptron(X, y, epochs, l_rate):
    weights = np.random.randn(y.shape[1], X.shape[1]) * 0.1 # Matriz com dimensões: num_classes X num_atributos
    
    for epoch in range(epochs): # Iterando épocas
        
        z = X @ weights.T
        result = activate_functions('softmax', z)
        error =  result - y # Erro por classe
        grad = error / len(X)

        # Ajustar os pesos para cada classe separadamente
        weights -= l_rate * np.dot(grad.T, X)

        if epoch % 5 == 0:
            print('Epoch: {}/{}'.format(epoch, epochs))
            
    print('Done!')
    
    return weights

# Treinar o classificador
weights = train_logistic_perceptron(X_train, y_train, epochs=3500, l_rate=0.01)
# print(weights.shape)

# Função de predição usando o classificador linear
def predict_logistic_perceptron(X, W):
    z = X @ weights.T
    result = activate_functions('softmax', z)

    # Converte as saídas para as classes preditas (0 a 9) usando a função argmax
    # A classe predita será o índice do valor máximo em cada linha
    classe = np.argmax(result, axis=1)

    return np.expand_dims(classe, axis=1)

# Realizar a predição no conjunto de teste
y_pred_test = predict_logistic_perceptron(X_test, weights)

print(y_pred_test.shape)
print(y_test.shape)

# Avaliar o desempenho do classificador
error = (len(y_test) - sum(y_pred_test == y_test)) / len(y_test) 
print("Error: {}".format(error[0]))
print("Accuracy: {}".format(1 - error[0]))
print(weights)