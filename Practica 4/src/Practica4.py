## Ramón Arjona Quiñones
## Celia Castaños Bornaechea

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.io import loadmat
from ML_utilities import forward_prop, trainNeutralNetwork, calcula_porcentaje
from displayData import displayData
from checkNNGradients import checkNNGradients


def Ejercicio1(lamda, num_iter):
    '''
    Redes neuronales
    '''
    # 1. Cargamos los datos
    data = loadmat('ex4data1.mat') 
    X = data['X'] # (5000x400)
    y = data['y'].ravel() #(5000,)
    y = (y - 1) #Porque están de 1 - 10 y los queremos del 0 - 9
    m = X.shape[0]

    # 2. Atributos de la red neuronal
    num_entradas = 400
    num_ocultas = 25
    num_etiquetas = 10

    # 3. Inicializamos y_onehot
    y_onehot = np.zeros((m, num_etiquetas)) #5000 x 10
    for i in range(m):
        y_onehot[i][y[i]] = 1

    # Visualizar 100 ejemplos
    #sample = np.random.choice(X.shape[0], 100)
    #fig, ax = displayData(X[sample])
    #plt.show()

    # 4. Entrenamos la red neuronal y sacamos los pesos óptimos
    theta1, theta2 = trainNeutralNetwork(num_entradas, num_ocultas, num_etiquetas, X, y_onehot, lamda, num_iter)

    # 5. Con los pesos óptimos obtenidos, hacemos la propagación hacia delante y obtenemos la predicción de la red
    unosX = np.hstack([np.ones([m, 1]), X])
    a1, z2, a2, z3, h = forward_prop(unosX, theta1, theta2) 

    # Sacamos el porcentaje de aciertos
    porcentaje = calcula_porcentaje(y, h, 3)
    print("La red clasificado bien un",  porcentaje, " % de los ejemplos, con λ = ", lamda, " y ",  num_iter, " iteraciones")

# lamda = 1, 70 iteraciones
Ejercicio1(1, 70)
Ejercicio1(1, 50)
Ejercicio1(1, 100)
Ejercicio1(3, 70)
Ejercicio1(3, 50)
Ejercicio1(3, 100)