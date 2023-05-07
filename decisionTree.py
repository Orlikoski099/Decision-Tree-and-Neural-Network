from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt
from readFile import readFile as reader
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt



def training(label=0):
    base = reader(label)
    vetor = []

    for b in base:
        values = np.ravel(list(b.values()))
        vetor.append(values)

    if(label > 0):
        df = pd.DataFrame(vetor, columns=['qPA', 'bpm', 'fpm', 'gravidade'])
        X = df[['qPA', 'bpm', 'fpm']]
        y = df['gravidade']

        # print(df)

        df['gravidade'] = pd.cut(df['gravidade'], bins=[-1, 25, 50, 75, 100],
                                  labels=['critico', 'instavel', 'potencialmente estavel', 'estavel'], include_lowest=True)

        df['gravidade'].fillna(df['gravidade'].mode()[0], inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(df.drop('gravidade', axis=1), df['gravidade'], test_size=0.2)

    else:
        df = pd.DataFrame(vetor, columns=['qPA', 'bpm', 'fpm', 'gravidade'])
        X = df[['qPA', 'bpm', 'fpm']]
        y = df['gravidade']

        df['gravidade'] = pd.cut(df['gravidade'], bins=[-1, 25, 50, 75, 100],
                                  labels=['critico', 'instavel', 'potencialmente estavel', 'estavel'], include_lowest=True)


        df['gravidade'].fillna(df['gravidade'].mode()[0], inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(df.drop('gravidade', axis=1), df['gravidade'], test_size=0.5)

    clf = DecisionTreeClassifier(random_state=10)
    clf.fit(X_train, y_train)

    # Faça previsões no conjunto de teste e avalie o desempenho do modelo
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Acurácia:', accuracy)

    r = export_text(clf, feature_names=['qPA', 'bpm', 'fpm'])
    with open('tree.txt', 'w') as f:
        f.write(r)

    plt.figure(figsize=(20, 10), facecolor='k')
    plot_tree(clf,
              feature_names=['qPA', 'bpm', 'fpm'],
              class_names=['critico', 'instavel', 'potencialmente estavel', 'estavel'],
              rounded=True,
              filled=True,
              fontsize=14)
    plt.savefig('arvore.png')

    cm = confusion_matrix(y_test, y_pred)

    # Exiba a matriz de confusão em forma de tabela
    cmd = ConfusionMatrixDisplay(cm, display_labels=['critico', 'instavel', 'potencialmente estavel', 'estavel'])
    cmd.plot()
    plt.savefig('matriz-confusão.png')

    def predict_gravidade(X):
        classes = clf.predict(X)
        probas = clf.predict_proba(X)
        gravidade = (probas[:,1]*25 + probas[:,2]*50 + probas[:,3]*75)
        return gravidade[0], classes[0]

    return predict_gravidade


model = training()

samples = np.array([[-5.5577, 180.8004, 20.0000], 
                    [8.4034, 30.0393, 4.8394], 
                    [0.0399, 120.4928, 8.9384]])

for sample in samples:
    gravidade, classe = model(sample.reshape(1,-1))
    # print(f"{gravidade:.4f}, {classe}")
    print(gravidade, classe)