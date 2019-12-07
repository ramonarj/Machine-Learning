## Ramón Arjona Quiñones
## Celia Castaños Bornaechea

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
from sklearn.preprocessing import PolynomialFeatures 
from scipy.io import loadmat
from ML_utilities import h, hMatrix, sigmoid, regularizedCost, regularizedGradient
from displayData import displayData

def calcula_porcentaje(Y, Z, digitsNo: int):
    '''
    Calcula el porcentaje de aciertos del entrenador
    '''

    m = Y.shape[0]

    # Creamos la matriz
    results = np.empty(m)

    # Recorremos todos los ejemplos de entrenamiento...
    for i in range (m):
        results[i] = np.argmax(Z[i])
    results = results.T

    # Vemos cuántos de ellos coinciden con Y 
    Y = np.where (Y == 10, 0, Y) # Para que los '10' se cambien a '0'
    coinciden = ( Y == results )
    aciertos = np.sum(coinciden)

    # Porcentaje sobre el total de ejemplos redondeado
    return round((aciertos / m) * 100, digitsNo)

#Nº nodos en cada capa: 400, 25, 10
def forward_prop_generic(X, thetas, num_layers:int):
    m = X.shape[0]
    inputLayer = X

    # 1 iteración por cada capa de la matriz
    for i in range (num_layers):
        # Añadimos la columna de 1's a la entrada
        inputLayer = np.hstack([np.ones([m, 1]), inputLayer])
        # Calculamos la capa de salida
        outputLayer = hMatrix(inputLayer, thetas[i]) 
        # Capa de entrada de la siguiente iteración
        inputLayer = sigmoid(outputLayer)

    return inputLayer

#Nº nodos en cada capa: 400, 25, 10
def forward_prop(X, theta1, theta2):
    '''
    Propagación hacia delante en la red neuronal
    '''
    m = X.shape[0]

    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)

    return a1, z2, a2, z3, h

def back_prop (params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    '''
    Propagación inversa en la red neuronal para hallar el coste y el gradiente
    '''
    #theta1 = np.reshape (params_rn[:num_ocultas ∗ (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    #theta2 = np.reshape (params_rn[num_ocultas ∗ (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))



    #Hay que devolver:
    # 1) Coste 
    # 2) Gradiente


def Ejercicio1(lamda):
    '''
    Redes neuronales
    '''
    #Cargamos los datos
    data = loadmat('ex4data1.mat') 
    X = data['X'] # (5000x400)
    y = data['y'].ravel()
    num_etiquetas = 10

    # Visualizar los 100 ejemplos
    sample = np.random.choice(X.shape[0], 100)
    fig, ax = displayData(X[sample])
    plt.show()



#np.set_printoptions(threshold=sys.maxsize) #Para que escriba todos los valores de los arrays
Ejercicio1(0.1)