import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             explained_variance_score, max_error,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                          export_text, plot_tree)

from readFile import readFile as reader

TEST_SIZE = 0.2
RANDOM_STATE = 37

def training(label=0):
    base = reader(label)
    vetor = []

    for b in base:
        values = np.ravel(list(b.values()))
        vetor.append(values)

    def classification():
        df = pd.DataFrame(vetor, columns=['qPA', 'bpm', 'fpm', 'gravidade'])

        df['gravidade'] = pd.cut(df['gravidade'], bins=[0, 25, 50, 75, 100],
                                 labels=['1', '2', '3', '4'], include_lowest=True)

        X = df[['qPA', 'bpm', 'fpm']]
        y = df['gravidade']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        modelo = DecisionTreeClassifier(random_state=RANDOM_STATE)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)

        report = classification_report(y_test, y_pred, zero_division=1)

        print('\n\n--CLASSIFIER--\n')

        print(report)

        r = export_text(modelo, feature_names=['qPA', 'bpm', 'fpm'])
        with open('decision_tree.txt', 'w') as f:
            f.write(r)

        plt.figure(figsize=(130, 60), facecolor='k')
        plot_tree(modelo,
                  feature_names=['qPA', 'bpm', 'fpm'],
                  class_names=['Crítico', 'Instável', 'p. Estável', 'Estável'],
                  rounded=True,
                  filled=True,
                  node_ids=True,
                  impurity=False,
                  fontsize=14)
        plt.savefig('decision_tree.png')

        cm = confusion_matrix(y_test, y_pred)

        cmd = ConfusionMatrixDisplay(
            cm, display_labels=['Crítico', 'Instável', 'p. Estável', 'Estável'])
        cmd.plot()
        plt.savefig('decision_tree-matriz_confusão.png')

        def predict_gravidade(X):
            test = pd.DataFrame(X)
            test.columns = ['qPA', 'bpm', 'fpm']
            classes = modelo.predict(test)
            return classes[0]

        return predict_gravidade

    def regression():
        df = pd.DataFrame(vetor, columns=['qPA', 'bpm', 'fpm', 'gravidade'])
        X = df[['qPA', 'bpm', 'fpm']]
        y = df['gravidade']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        modelo = DecisionTreeRegressor(random_state=RANDOM_STATE)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        max_err = max_error(y_test, y_pred)

        print('\n--REGRESSÃO--\n')

        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Explained Variance Score: {evs:.4f}")
        print(f"Max Error: {max_err:.4f}")

        def predict_gravidade(X):
            test = pd.DataFrame(X)
            test.columns = ['qPA', 'bpm', 'fpm']
            gravidade = modelo.predict(test)
            return gravidade[0]

        return predict_gravidade

    return regression(), classification()


def decisionTree(samples):

    reg, cla = training()

    samples = np.array(samples)
    response = []
    i = 0

    for sample in samples:
        gravidade = reg(sample.reshape(1, -1))
        classe = cla(sample.reshape(1, -1))
        response.append([i, gravidade, classe])
        i += 1

    with open('decision_tree-test_response.txt', 'w') as f:
        for r in response:
            f.write(f'{r[0]}, {r[1]}, {r[2]}\n')
