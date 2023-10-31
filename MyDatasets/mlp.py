import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('real/lid/data.csv')

# Separando o dataset por classe
X = np.array(dataset.iloc[:, 1:-1])
y = np.array(dataset.iloc[:, -1])

# Separando dados de treino/teste
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, stratify=y, test_size=0.2)

print(X_train_raw.shape, y_train_raw.shape)
print(X_test_raw.shape, y_test_raw.shape)

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

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Funções de ativação para o neurônio
def activate_functions(type, matrix):
    if type == 'sigmoid':
        return 1 / (1 + np.exp(-matrix))
    elif type == 'softmax':
        exp_matrix = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
        return exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)   
    elif type == 'tanh':
        return np.tanh(matrix)

# Função de treino do MLP
def mlp_train(X, y, n_neurons_hlayer, epochs, l_rate, criteria):
    n_classes = y.shape[1]
    n_features = X.shape[1]

    w_input = np.random.randn(n_features, n_neurons_hlayer) * 0.1
    w_output = np.random.randn(n_neurons_hlayer, n_classes) * 0.1

    bias_input = np.random.randn(n_neurons_hlayer, 1) * 0.5
    bias_output = np.random.randn(n_classes, 1) * 0.5

    loss_history = []
    for epoch in range(epochs):
        
        # Forward
        Zin = (X @ w_input) + bias_input.T
        result_in = activate_functions('sigmoid', Zin)

        Zout = (result_in @ w_output) + bias_output.T
        result_out = activate_functions('softmax', Zout)

        # Backpropagation
        error_out = result_out - y
        grad_out = error_out / len(X)

        error_in = grad_out @ w_output.T
        grad_in = error_in * result_in * (1 - result_in)

        # Ajustar os pesos e os viéses
        w_input -= l_rate * np.dot(X.T, grad_in)
        w_output -= l_rate * np.dot(result_in.T, grad_out)

        bias_input -= l_rate * np.sum(grad_in, axis=0, keepdims=True).T
        bias_output -= l_rate * np.sum(grad_out, axis=0, keepdims=True).T

        if epoch == 0:
            loss = np.mean((y - result_out)**2)
            print('Initial Epoch: {}, loss: {}'.format(epoch, loss))
            best_loss = loss
            loss_history.append(loss)
            
        if epoch != 0 and epoch % 5 == 0:
            loss = np.mean((y - result_out)**2)
            print('Epoch: {}, loss: {}'.format(epoch, loss))
            loss_history.append(loss)

            if loss < best_loss:
                best_loss = loss
                wait = 0
            else: wait += 1

            if wait >= criteria:
                print('Final Epoch (loss stopped): {}, loss: {}'.format(epoch, loss))
                return loss_history, w_input, w_output, bias_input, bias_output

    loss = np.mean((y - result_out)**2)
    print('Last Epoch: {}, loss: {}'.format(epoch+1, loss))
                                      
    return loss_history, w_input, w_output, bias_input, bias_output

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

def random_search(X, y):
    mlp = MLPClassifier(activation='logistic', learning_rate_init=0.1) # Mantendo configurações do MLP - l_rate, função de ativação sigmoid, etc

    params_search = {"hidden_layer_sizes": list(np.arange(2,500))}
    
    search = RandomizedSearchCV(mlp, param_distributions=params_search, n_jobs=-1, cv=3, verbose=5) # Busca do melhor numero de neuronios da camada
    search.fit(X, y)
    best = search.best_params_['hidden_layer_sizes']
    
    return best

n_neurons = random_search(X_train, y_train)
print("Neurons in hidden layer: ", n_neurons)

loss_history, w_input, w_output, bias_input, bias_output = mlp_train(X_train, y_train, n_neurons, epochs=10000, l_rate=0.1, criteria=15)

# plt.plot(loss_history, color='red', label='Predict')
# plt.title('Loss History')
# plt.legend()
# # plt.xlim(0, 120)
# # plt.ylim(0, 120)
# plt.show()
# Realizar a predição no conjunto de teste
# Função de predição usando o classificador linear

def predict_logistic_perceptron(X, W):
    z = X @ weights.T
    result = activate_functions('softmax', z)

    # Converte as saídas para as classes preditas (0 a 9) usando a função argmax
    # A classe predita será o índice do valor máximo em cada linha
    classe = np.argmax(result, axis=1)

    return np.expand_dims(classe, axis=1)
y_pred_test = predict_logistic_perceptron(X_test, weights)

print(y_pred_test.shape)
print(y_test.shape)


# Avaliar o desempenho do classificador
y_pred_test = loss_history
print(y_pred_test)
error = (len(y_test) - sum(y_pred_test == y_test)) / len(y_test) 
print("Error: {}".format(error[0]))
print("Accuracy: {}".format(1 - error[0]))