from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from readFile import readFile as reader
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

base = reader(0)
vetor = []

for b in base:
    values =  np.ravel(list(b.values()))
    vetor.append(values)

df = pd.DataFrame(vetor, columns=['qPA', 'bpm', 'fpm', 'gravidade'])
X = df[['qPA', 'bpm', 'fpm']]
y = df['gravidade']

# print(df)

df['gravidade'] = pd.cut(df['gravidade'], bins=[-1, 25, 50, 75, 100], labels=['critico', 'instavel', 'potencialmente estavel', 'estavel'], include_lowest=True)

df['gravidade'].fillna(df['gravidade'].mode()[0], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df.drop('gravidade', axis=1), df['gravidade'], test_size=0.2)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Faça previsões no conjunto de teste e avalie o desempenho do modelo
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Acurácia:', accuracy)


plt.figure(figsize=(20, 10), facecolor='k')
plot_tree(clf, 
          feature_names=['qPA', 'bpm', 'fpm'], 
          class_names=['critico', 'instavel', 'potencialmente estavel', 'estavel'], 
          rounded=True, 
          filled=True,
          fontsize=14)
plt.show()

cm = confusion_matrix(y_test, y_pred)

# Exiba a matriz de confusão em forma de tabela
cmd = ConfusionMatrixDisplay(cm, display_labels=['critico', 'instavel', 'potencialmente estavel', 'estavel'])
cmd.plot()
plt.show()
