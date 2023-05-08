import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, mean_squared_error, r2_score)
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

        df['gravidade'] = pd.cut(df['gravidade'], bins=[-1, 25, 50, 75, 100],
                                 labels=['1', '2', '3', '4'], include_lowest=True)

        X_train, X_test, y_train, y_test = train_test_split(df.drop('gravidade', axis=1), df['gravidade'], test_size=0.5, random_state=32)

        X = df[['qPA', 'bpm', 'fpm']]
        y = df['gravidade']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        modelo = DecisionTreeClassifier(random_state=RANDOM_STATE)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('Acurácia:', accuracy)

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

        print('MSE: ', mean_squared_error(y_test, y_pred))
        print('R²: ', r2_score(y_test, y_pred))

        def predict_gravidade(X):
            test = pd.DataFrame(X)
            test.columns = ['qPA', 'bpm', 'fpm']
            gravidade = modelo.predict(test)
            return gravidade[0]

        return predict_gravidade

    return regression(), classification()


reg, cla = training()

samples = []

for i in range(100):
    qPA = round(random.uniform(-10, 10), 4)
    bpm = round(random.uniform(0, 200), 4)
    fpm = round(random.uniform(0, 22), 4)
    samples.append([qPA, bpm, fpm])

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
