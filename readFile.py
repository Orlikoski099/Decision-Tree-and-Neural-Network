COM_LABEL = 'D:/facul/SI/Decision-Tree-and-Neural-Network/treino_sinais_vitais_com_label.txt'
SEM_LABEL = 'D:/facul/SI/Decision-Tree-and-Neural-Network/treino_sinais_vitais_sem_label.txt'
# D:\facul\SI\Decision-Tree-and-Neural-Network\treino_sinais_vitais_sem_label.txt
class Sinais:
    def __init__(self, id, pSist, pDiast, qPA, bpm, fpm, gravidade, label = None):
        self.id = id                                      #Identificação da vítima
        self.pSist = pSist                                #Pressão sistólica
        self.pDiast = pDiast                              #Pressão diastólica
        self.qPA = qPA                                    #Qualidade da pressão
        self.bpm = bpm                                    #Pulso ou Batimentos por Minuto
        self.fpm = fpm                                    #Frequência da respiração
        self.gravidade = gravidade                        #Gravidade
        self.label = label                                #Grau de gravidade (1 - CRÍTICO | 2 - INSTÁVEL | 3 - POTENCIALMENTE ESTÁVEL | 4 - ESTÁVEL)

    def values(self):
        return [self.qPA, self.bpm, self.fpm, self.gravidade]
pessoas = []

def readFileWihLabel(filename):
    with open(filename, 'r') as file:
        for line in file:
            id, pSist, pDiast, qPA, bpm, fpm, gravidade, label = map(float, line.strip().split(','))
            pessoa = Sinais(id, pSist, pDiast, qPA, bpm, fpm, gravidade, label)
            pessoas.append(pessoa)
    return pessoas 

def readFileWihoutLabel(filename):
    with open(filename, 'r') as file:
        for line in file:
            id, pSist, pDiast, qPA, bpm, fpm, gravidade = map(float, line.strip().split(','))
            pessoa = Sinais(id, pSist, pDiast, qPA, bpm, fpm, gravidade)
            pessoas.append(pessoa)
    return pessoas

#Lê os arquivos, por padrão lê o com a Label
def readFile(withLabel = 1):
    if withLabel > 0:
        return readFileWihLabel(COM_LABEL)
    return readFileWihoutLabel(SEM_LABEL)