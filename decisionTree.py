from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from readFile import readFile as reader
import pandas as pd
import numpy as np

base = reader(0)
vetor = []

for b in base:
    values =  np.ravel(list(b.values()))
    vetor.append(values)

df = pd.DataFrame(vetor, columns=['qualidade de pressão', 'BPM', 'frequência de respiração', 'gravidade'])

X = df.drop(['gravidade'], axis=1)
y = df['gravidade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Acurácia:', accuracy)