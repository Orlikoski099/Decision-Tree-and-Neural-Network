import random
from randomForest import randomForest
from decisionTree import decisionTree
from neuralNetwork import neuralNetwork


#Preencha para setar um arquivo de teste cego e utiliza-lo nos testes.
FILENAME = ''

samples = []

#Trecho abaixo usado para gerar um arquivo de teste cego com valores aleatórios dentro das margens permitidas.
for i in range(100):
    qPA = round(random.uniform(-10, 10), 4)
    bpm = round(random.uniform(0, 200), 4)
    fpm = round(random.uniform(0, 22), 4)
    samples.append([qPA, bpm, fpm])


#Trecho abaixo usado para carregar um arquivo de teste cego pré moldado. Se usar o abaixo comentar o de cima.
# with open(FILENAME, 'r') as file:
#     for line in file:
#         id, qPA, bpm, fpm = map(float, line.strip().split(','))
#         samples.append([qPA, bpm, fpm])

def test(samples):
    print('\n\n\n---DECISION TREE---\n')
    decisionTree(samples)
    print('\n\n\n---RANDOM FOREST---\n')
    randomForest(samples)
    print('\n\n\n---NEURAL NETWORK---\n')
    neuralNetwork(samples)

test(samples)