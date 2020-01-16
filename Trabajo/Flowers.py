## Ramón Arjona Quiñones
## Celia Castaños Bornaechea

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import random as rn

from tqdm import tqdm
from scipy.optimize import fmin_tnc
from sklearn.preprocessing import PolynomialFeatures 
from scipy.io import loadmat
from ML_utilities import h, hMatrix, sigmoid, regularizedCost, regularizedGradient

X=[]
Y=[]

IMG_SIZE = 150
RES_PATH = 'flowers/'
FLOWER_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
FLOWER_COUNT = [769, 1055, 784, 734, 984]

        
def LoadFlowerImages(flowerType):
    '''
    Carga las imágenes del directorio especificado y rellena la Y (one_hot) con el índice especificado
    '''
    folder = RES_PATH + FLOWER_NAMES[flowerType]
    for img in tqdm(os.listdir(folder)): #Recorre las imágenes de ese directorio
        # label = assign_label(img,flower_type)
        path = os.path.join(folder,img) #Path de la imagen
        img = cv2.imread(path, cv2.IMREAD_COLOR) #Lee la imagen
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) #La reescala
        row = []
        for i in range(0, 5):
            if i==flowerType:
                row.append(1)
            else:
                row.append(0)
        
        X.append(np.array(img)) #Añade la imagen a los casos de entrenamiento
        Y.append(row) #Pone en la posición correspondiente de la matriz salida qué flor es 
        


def main():

    #Cargamos todas las imágenes de sus respectivas carpetas
    for i in range (2): #len(FLOWER_NAMES)
        LoadFlowerImages(i)

    print(Y[760])
    print(Y[FLOWER_COUNT[0]])


    # PRUEBA INI - carga imagen y la muestra
    # img=mpimg.imread(RES_PATH+FLOWER_NAMES[0])
    # imgplot = plt.imshow(img)
    # plt.show()


main()