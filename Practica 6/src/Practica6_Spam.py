## Ramón Arjona Quiñones
## Celia Castaños Bornaechea

import numpy as np
from ML_utilities import getBestSVMModel, calcula_porcentaje_Y
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from process_email import email2TokenList
from get_vocab_dict import getVocabDict

X = []
y = []

#3301 ejemplos
NUM_SPAM = 500
NUM_HARD_HAM = 250
NUM_EASY_HAM = 2551


def MakeTrainData(folder, size, Yvalue):
    '''
    Lee los archivos de la carpeta correspondiente, los guarda en las X
    y pone el valor de las Y correspondiente
    '''
     #Leemos los de spam
    for i in range(size):
        email_contents = open('{0}/{1:04d}.txt'.format(folder, i+1), 'r', encoding='utf-8', errors='ignore').read()
        email = email2TokenList(email_contents)

        # Palabras del diccionario
        vocab = getVocabDict()
        words = np.zeros([len(vocab)], dtype=int) #Será la X

        # Vemos cuales aparecen en el correo y rellenamos con 1's y 0's
        for i in range (len(email)):
            token = email[i]
            # Comprobamos que no es una palabra incompleta o mal escrita
            if token in vocab:
                pos = vocab[token] #Posición que ocupa la palabra en el diccionario
                words[pos - 1] = 1 #Ponemos esa posición a 1 (empezando en 0)
        X.append(words)
        y.append(Yvalue)

def SpamDetector(values):
    '''
    Detector de spam usando SVM
    '''
    # 1. Leemos los correos y los añadimos a la X e Y
    # Spam
    MakeTrainData("spam", NUM_SPAM, 1)
    # No spam
    MakeTrainData("easy_ham", NUM_EASY_HAM, 0)
    MakeTrainData("hard_ham", NUM_HARD_HAM, 0)

    # 2. Hacemos la DIVISIÓN de entrenamiento, validación y test
    #Dividimos en entrenamiento / test (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(y), test_size=0.2, random_state=42)
    #Volvemos a dividir en entrenamiento / validación, para tener un total de 60% (train), 20% (val), 20% (test)
    X_train, X_val, y_train, y_val = train_test_split(np.asarray(X_train), np.asarray(y_train), test_size=0.25, random_state=42)
    # 2217, 660, 661

    # 3. ENTRENAMOS el SVM y nos quedamos con los parámetros que den mejor resultado sobre el conjunto de validación
    svm, C, sigma = getBestSVMModel('rbf', values, X_train, y_train, X_val, y_val)

    # 4. Vemos el porcentaje sobre los ejemplos de test
    h = svm.predict(X_test)
    porc = calcula_porcentaje_Y(y_test.ravel(), h, 4)

    print("Precisión del clasificador de spam: " + str(porc) + "%, C=" + str(C) + ", σ=" + str(sigma))

# Lista de posibles valores para C y sigma 
#values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
values = [0.1, 0.3, 1] # Que si no tarda 3 años y medio (optimos: C = 0.1, sigma = 1)
SpamDetector(values)