## Ramón Arjona Quiñones
## Celia Castaños Bornaechea

# numpy #
import numpy as np
# matplotlib #
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# opencv #
import cv2
# sistema operativo #
import os
# barra de progreso #
from tqdm import tqdm
# sklearn #
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
# nuestro #
from ML_utilities import trainNeutralNetwork, forward_prop, calcula_porcentaje, calcula_porcentaje_Y, sigmoid, hMatrix, oneVsAll, makeOneHot

#Atributos
IMG_SIZE = 32 #Ponerlo a 32 o 64
RES_PATH = 'flowers/'
FLOWER_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
FLOWER_COUNT = [0, 0, 0, 0, 0] #Se rellena solo


# DATOS
X = []
y = []
onehotY = []

# PORCENTAJES PARA DIVIDIR EL DATASET
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2
TEST_FRACTION = 0.2

##TODO: leer las imágenes en color, porque si no nos vamos a comer un mojón de precisión

def LoadAllImages():
    '''
    Carga todas las imágenes
    '''
    for i in range (len(FLOWER_NAMES)):
        LoadFlowerImages(i)

def LoadFlowerImages(flowerType):
    '''
    Carga las imágenes del directorio especificado y rellena la Y (one_hot) con el índice especificado
    '''
    folder = RES_PATH + FLOWER_NAMES[flowerType]
    indice = 0
    #Calculamos el índice del 1er elemento que vamos a meter
    for i in range(flowerType):
        indice+=FLOWER_COUNT[i]

    #Recorre las imágenes de ese directorio
    for img in tqdm(os.listdir(folder)): 
        # label = assign_label(img,flower_type)
        path = os.path.join(folder,img) #Path de la imagen
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #Lee la imagen
        new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) #La reescala a SIZEXSIZE
        row = []
        for i in range(0, len(FLOWER_NAMES)):
            if i==flowerType:
                row.append(1)
            else:
                row.append(0)
     
        X.append(np.array(new_array / 255).ravel()) #Añade la imagen a los casos de entrenamiento
        y.append(flowerType)
        onehotY.append(row) #Pone en la posición correspondiente de la matriz salida qué flor es 
        FLOWER_COUNT[flowerType] += 1 #Añadimos una flor al recuento

def DivideSets(X, y):
    '''
    Divide el dataset en 3 sets de entrenamiento, validación y test
    '''
    #Dividimos en entrenamiento / test (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(y), test_size=0.2, random_state=42)
    #Volvemos a dividir en entrenamiento / validación, para tener un total de 60% (train), 20% (val), 20% (test)
    X_train, X_val, y_train, y_val = train_test_split(np.asarray(X_train), np.asarray(y_train), test_size=0.25, random_state=42)

    #Devolvemos una tupla con los conjuntos troceados
    return [np.asarray(X_train), np.asarray(y_train), np.asarray(X_val), np.asarray(y_val), np.asarray(X_test), np.asarray(y_test)]
        

# Regresión logística multiclase #
def LogisticRegressionClassifier(datasets:tuple, lamda):
    '''
    Clasificador por regresión logística
    '''
    # Atributos
    num_etiquetas = len(FLOWER_NAMES)
    X_train = datasets[0]
    y_train = datasets[1]
    X_test = datasets[4]
    y_test = datasets[5]

    # Resolvemos el one vs All con el conjunto de entrenamiento
    m = X_train.shape[0]
    unosX = np.hstack([np.ones([m, 1]), X_train])
    thetas = oneVsAll(unosX, np.asarray(y_train), num_etiquetas, lamda)

    # Vemos su precisión sobre el conjunto de tests
    m = X_test.shape[0]
    unosX = np.hstack([np.ones([m, 1]), X_test])
    z = sigmoid(hMatrix(unosX, thetas))

    print("El OneVsAll tiene una precisión del ", calcula_porcentaje(y_test, z, 4), "%")


# Red neuronal de 2 capas #
def NeutralNetworkClassifier(datasets:tuple, lamda:float, num_ocultas:int, num_iter:int):
    '''
    Clasificador por red neuronal de 2 capas 
    Recibe el número de ocultas, el término de regularización 
    y las iteraciones máximas en el descenso de gradiente
    '''
    # 1. Montamos la red neuronal
    # Atributos
    num_entradas = IMG_SIZE * IMG_SIZE # 1 por cada píxel
    # num_ocultas = nos la pasan como parámetro
    num_etiquetas = len(FLOWER_NAMES) # 5

    #Datasets
    X_train = datasets[0]
    y_train_onehot = makeOneHot(datasets[1], num_etiquetas)
    X_test = datasets[4]
    y_test = datasets[5]
    m = X_train.shape[0]

    # La entrenamos y cogemos los pesos óptimos (es el código de la Práctica 4)
    print("··· Entrenando la red neuronal (puede tardar varios minutos) ··· ")
    theta1, theta2 = trainNeutralNetwork(num_entradas, num_ocultas, num_etiquetas, X_train, y_train_onehot, lamda, num_iter)

    # 2. Con los pesos óptimos obtenidos, hacemos la propagación hacia delante y obtenemos la predicción de la red
    print("··· Comprobando la precisión sobre el conjunto de test ··· ")
    m = X_test.shape[0]
    unosX = np.hstack([np.ones([m, 1]), X_test])
    a1, z2, a2, z3, h = forward_prop(unosX, theta1, theta2) 

    # Sacamos el porcentaje de aciertos
    porcentaje = calcula_porcentaje(y_test, h, 3)
    print("La red tiene una precisión del ",  porcentaje, " %")

# Support Vector Machines #
def SVMClassifier(kernelType:str, reg:float, sigma:float):

    #Datasets
    X_train = datasets[0]
    y_train = datasets[1]
    X_test = datasets[4]
    y_test = datasets[5]
    

    # Hacemos el SVM con kernel especificado
    if(kernelType == 'linear'):
        svm = SVC(kernel='linear', C=reg)
    elif (kernelType == 'rbf'):
        svm = SVC(kernel='rbf', C=reg, gamma=1 / (2 * sigma ** 2))

    # Hacemos que se ajuste a los datos de entrenamiento
    print("··· Entrenando la SVM (puede tardar varios minutos) ··· ")
    svmMultiClass = OneVsRestClassifier(svm) # de sklearn
    svmMultiClass.fit(X_train, y_train)

    # Falta hacer validación
    
    #Vemos el porcentaje de aciertos sobre el conjunto de tests
    print("··· Comprobando la precisión sobre el conjunto de test ··· ")
    h = svmMultiClass.predict(X_test)
    porcentaje = calcula_porcentaje_Y(y_test, h, 4)

    print("La SVM tiene una precisión del ",  porcentaje, " %")


# 1. Cargamos todas las imágenes de sus respectivas carpetas
print("··· Cargando las imágeners de las flores ··· ")
LoadAllImages()
# 2. Dividimos los ejemplos en 3 sets: entrenamiento, validación y test
datasets = DivideSets(X, y)
# 3. Regresión logística
#LogisticRegressionClassifier(datasets, 0.1)
# 4. Red neuronal
#NeutralNetworkClassifier(datasets, 1, 100, 140)
# 5. SVM
#values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
SVMClassifier('rbf', 0.1, 1)
