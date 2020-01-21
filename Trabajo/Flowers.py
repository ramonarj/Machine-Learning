## Ramón Arjona Quiñones
## Celia Castaños Bornaechea

# numpy # (para todo)
import numpy as np
# matplotlib # (para dibujar)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# sklearn # (para SVM y para dividir los conjuntos de datos)
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
# scipy # (para cargar y guardar matrices)
from scipy.io import savemat, loadmat
# opencv # (para leer imágenes)
import cv2
# sistema operativo # (para los paths)
import os
# tqdm # (para la barra de progreso)
from tqdm import tqdm
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

# PORCENTAJES PARA DIVIDIR EL DATASET (60-20.20)
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2
TEST_FRACTION = 0.2

DATA_FILENAME = "flowersData.mat"

##TODO: leer las imágenes en color, porque si no nos vamos a comer un mojón de precisión

def LoadAllImages():
    '''
    Carga todas las imágenes
    '''
    print("··· Cargando las imágeners de las flores ··· ")
    for i in range (len(FLOWER_NAMES)):
        LoadFlowerImages(i)

    # Devolvemos los sets ya divididos
    return DivideSets(X, y)

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
        img = cv2.imread(path, cv2.IMREAD_COLOR) #Lee la imagen
        new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) #La reescala a SIZEXSIZE

        #Añade la imagen a los casos de entrenamiento
        X.append(np.array(new_array / 255).ravel()) 
        y.append(flowerType)
        FLOWER_COUNT[flowerType] += 1 #Añadimos una flor al recuento


def DivideSets(X, y):
    '''
    Divide el dataset en 3 sets de entrenamiento, validación y test
    '''
    #Dividimos en entrenamiento / test (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(y), test_size=TEST_FRACTION, random_state=42)
    #Volvemos a dividir en entrenamiento / validación, para tener un total de 60% (train), 20% (val), 20% (test)
    X_train, X_val, y_train, y_val = train_test_split(np.asarray(X_train), np.asarray(y_train), test_size=0.25, random_state=42)
    #X_mock, X_val, y_mock, y_val = train_test_split(np.asarray(X_train), np.asarray(y_train), test_size=0.25, random_state=42)

    #Devolvemos una tupla con los conjuntos troceados
    return [np.asarray(X_train), np.asarray(y_train), np.asarray(X_val), np.asarray(y_val), np.asarray(X_test), np.asarray(y_test)]
        

# Regresión logística multiclase #
def LogisticRegressionClassifier(data:dict, lamda):
    '''
    Clasificador por regresión logística
    '''
    # Atributos
    num_etiquetas = len(FLOWER_NAMES)
    
    # Sacamos los conjuntos del diccionario
    X_train = data["X_train"]
    y_train = data["y_train"].ravel()
    X_test = data["X_test"]
    y_test = data["y_test"].ravel()

    # Resolvemos el one vs All con el conjunto de entrenamiento
    print("··· Haciendo el descenso de gradiente ··· ")
    m = X_train.shape[0]
    unosX = np.hstack([np.ones([m, 1]), X_train])
    thetas = oneVsAll(unosX, np.asarray(y_train), num_etiquetas, lamda)

    # Vemos su precisión sobre el conjunto de tests
    m = X_test.shape[0]
    unosX = np.hstack([np.ones([m, 1]), X_test])
    z = sigmoid(hMatrix(unosX, thetas))

    print("-> El OneVsAll tiene una precisión del ", calcula_porcentaje(y_test, z, 4), "% <-")


# Red neuronal de 2 capas #
def NeutralNetworkClassifier(data:dict, lamda:float, num_ocultas:int, num_iter:int):
    '''
    Clasificador por red neuronal de 2 capas 
    Recibe el número de ocultas, el término de regularización 
    y las iteraciones máximas en el descenso de gradiente
    '''
    # 1. Montamos la red neuronal
    # Atributos
    num_entradas = IMG_SIZE * IMG_SIZE * 3 # 3 por cada píxel (R, G, B)
    # num_ocultas = nos la pasan como parámetro
    num_etiquetas = len(FLOWER_NAMES) # 5

    # Sacamos los datos del diccionario
    X_train = data["X_train"]
    y_train_onehot = makeOneHot(data["y_train"].ravel(), num_etiquetas)
    X_test = data["X_test"]
    y_test = data["y_test"].ravel()
    m = X_train.shape[0]

    # La entrenamos y cogemos los pesos óptimos (es el código de la Práctica 4)
    print("··· Entrenando la red neuronal (puede tardar varios minutos) ··· ")
    theta1, theta2 = trainNeutralNetwork(num_entradas, num_ocultas, num_etiquetas, X_train, y_train_onehot, lamda, num_iter)

    # 2. Con los pesos óptimos obtenidos, hacemos la propagación hacia delante y obtenemos la predicción de la red
    print("··· Comprobando la precisión sobre el conjunto de test ··· ")
    a1, z2, a2, z3, h = forward_prop(X_test, theta1, theta2) 

    # Sacamos el porcentaje de aciertos
    porcentaje = calcula_porcentaje(y_test, h, 3)
    print("-> La red tiene una precisión del ",  porcentaje, " % <-")

# Support Vector Machines #
def SVMClassifier(data:dict, kernelType:str, reg:float, sigma:float):

    # Sacamos los datos del diccionario
    X_train = data["X_train"]
    y_train = data["y_train"].ravel()
    X_test = data["X_test"]
    y_test = data["y_test"].ravel()
    
    # Hacemos el SVM con kernel especificado
    if(kernelType == 'linear'):
        svm = SVC(kernel='linear', C=reg)
    elif (kernelType == 'rbf'):
        svm = SVC(kernel='rbf', C=reg, gamma=1 / (2 * sigma ** 2))

    # Hacemos que se ajuste a los datos de entrenamiento
    print("··· Entrenando la SVM (puede tardar varios minutos) ··· ")
    svmMultiClass = OneVsRestClassifier(svm) # de sklearn
    svmMultiClass.fit(X_train, y_train)
    
    # Vemos el porcentaje de aciertos sobre el conjunto de tests
    print("··· Comprobando la precisión sobre el conjunto de test ··· ")
    h = svmMultiClass.predict(X_test)
    porcentaje = calcula_porcentaje_Y(y_test, h, 4)

    print("-> La SVM tiene una precisión del ",  porcentaje, "% <-")

def SaveSets(filename:str, datasets:tuple):
    '''
    Guarda los datasets ya separados en un archivo de MatLab 
    para no tener que cargar las imágenes todo el rato
    '''
    savemat(filename, mdict={
        'X_train': datasets[0], 
        'y_train': datasets[1], 
        'X_val': datasets[2], 
        'y_val': datasets[3], 
        'X_test': datasets[4],
        'y_test': datasets[5]})

    print("··· Datasets guardados en " + filename +  " ··· ")


# 1a). Cargamos todas las imágenes de sus respectivas carpetas y guardamos los datasets 
# (solo hay que llamar a esto una vez)
'''
datasets = LoadAllImages()
SaveSets(DATA_FILENAME, datasets)
'''
#values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30] #Para la validación 
# 1b). Cargamos los datasets ya preparados
data = loadmat(DATA_FILENAME)
# 2. Llamamos al clasificador que sea
#LogisticRegressionClassifier(data, 0.1)
#NeutralNetworkClassifier(data, 1, 25, 70)
SVMClassifier(data, 'rbf', 3, 1)