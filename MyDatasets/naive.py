import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def naive_bayes(data_path):
    # Carregue o conjunto de dados Iris
    print(data_path)
    dataset = pd.read_csv(data_path)
    data = np.array(dataset.iloc[:, 1:-1])
    target = np.array(dataset.iloc[:, -1])


    # Divida o conjunto de dados em treinamento (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Crie o classificador Naive Bayes Gaussiano
    nb_classifier = GaussianNB()

    # Ajuste o modelo aos dados de treinamento
    nb_classifier.fit(X_train, y_train)

    # Faça previsões no conjunto de teste
    y_pred = nb_classifier.predict(X_test)

    # Avalie a precisão do modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy:.2f}")

    # Exiba a matriz de confusão e o relatório de classificação
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Matriz de Confusão:")
    print(confusion)
    print("\nRelatório de Classificação:")
    print(report)

naive_bayes('real/iris/data.csv')
naive_bayes('real/ban/data.csv')
naive_bayes('real/car/data.csv')
