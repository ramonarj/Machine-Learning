## Ramón Arjona Quiñones
## Celia Castaños Bornaechea

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os

from tqdm import tqdm
from scipy.optimize import fmin_tnc
from sklearn.preprocessing import PolynomialFeatures 
from scipy.io import loadmat
from ML_utilities import h, hMatrix, sigmoid, regularizedCost, regularizedGradient

X=[]
Y=[]

IMG_SIZE = 20
DIR_DAISY='flowers/daisy'
DIR_SUNFLOWER='flowers/sunflower'
DIR_TULIP='flowers/tulip'
DIR_DANDI='flowers/dandelion'
DIR_ROSE='flowers/rose'

    

        
def LoadFlowerImages(DIR, flowerType):
    for img in tqdm(os.listdir(DIR)): #Recorre las imágenes de ese directorio
        # label = assign_label(img,flower_type)
        path = os.path.join(DIR,img) #Path de la imagen
        img = cv2.cv2.imread(path, cv2.cv2.IMREAD_COLOR) #Lee la imagen
        img = cv2.cv2.resize(img, (IMG_SIZE, IMG_SIZE)) #La reescala
        
        X.append(np.array(img)) #Añade la imagen a los casos de entrenamiento
        Y.append(flowerType) #Pone en la posición correspondiente de la matriz salida qué flor es 
        


def main():
    LoadFlowerImages(FLOWER_DAISY_DIR, 'Daisy' )
    print(len(X))
    # PRUEBA INI - carga imagen y la muestra
    # img=mpimg.imread(FLOWER_DAISY_DIR)
    # imgplot = plt.imshow(img)
    # plt.show()


main()