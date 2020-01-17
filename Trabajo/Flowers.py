## Ramón Arjona Quiñones
## Celia Castaños Bornaechea
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
from tqdm import tqdm

from ML_utilities import trainNeutralNetwork, forward_prop, calcula_porcentaje

IMG_SIZE = 20
RES_PATH = 'flowers/'
FLOWER_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
FLOWER_COUNT = [0, 0, 0, 0, 0] #Se rellena solo


X = []
Y = []
onehotY = []

        
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
        Y.append(flowerType)
        onehotY.append(row) #Pone en la posición correspondiente de la matriz salida qué flor es 
        FLOWER_COUNT[flowerType] += 1 #Añadimos una flor al recuento
        

#De momento lo estamos haciendo en escala de grises
def NeutralNetworkClassifier(lamda:float, num_ocultas:int, num_iter:int):

    # 1. Cargamos todas las imágenes de sus respectivas carpetas
    for i in range (len(FLOWER_NAMES)): #len(FLOWER_NAMES)
        LoadFlowerImages(i)

    # Necesitamos usar arrays de numpy
    nX = np.asarray(X)
    nY = np.asarray(onehotY)
    y = np.asarray(Y)

    m = nX.shape[0]


    # PRUEBA INI - carga imagen y la muestra
    # img=mpimg.imread(RES_PATH+FLOWER_NAMES[0])
    # imgplot = plt.imshow(img)
    # plt.show()

    # 2. Montamos la red neuronal
    # Atributos
    num_entradas = IMG_SIZE * IMG_SIZE # 1 por cada píxel
    # num_ocultas = nos la pasan como parámetro
    num_etiquetas = len(FLOWER_NAMES) # 5

    # La entrenamos y cogemos los pesos óptimos (es el código de la Práctica 4)
    theta1, theta2 = trainNeutralNetwork(num_entradas, num_ocultas, num_etiquetas, nX, nY, lamda, num_iter)

    # 3. Con los pesos óptimos obtenidos, hacemos la propagación hacia delante y obtenemos la predicción de la red
    unosX = np.hstack([np.ones([m, 1]), nX])
    a1, z2, a2, z3, h = forward_prop(unosX, theta1, theta2) 

    # Sacamos el porcentaje de aciertos
    porcentaje = calcula_porcentaje(y, h, 3)
    print("La red clasificado bien un",  porcentaje, " % de los ejemplos")


NeutralNetworkClassifier(1, 100, 140)