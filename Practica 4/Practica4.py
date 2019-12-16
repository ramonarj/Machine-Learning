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
    coinciden = ( Y == results )
    aciertos = np.sum(coinciden)

    # Porcentaje sobre el total de ejemplos redondeado
    return round((aciertos / m) * 100, digitsNo)

def network_cost(H, Y):
    '''
    Calcula el coste de manera vectorizada para la red neuronal,
    con una salida de la red H y la Y de los ejemplos de entrenamiento
    '''
    #Variables auxiliares
    m = Y.shape[0]

    # Usamos "multiply" en vez de "dot" para que haga multiplicación elemento a elemento, (no producto escalar)
    # y así luego los sumamos todos en vez de hacer un doble bucle
    ## Coste cuando Y = 1
    costeUno = np.multiply(Y, np.log(H)).sum() # Suma todos los elementos de la matriz (Y x H)
    ## Coste cuando Y = 0
    costeCero = np.multiply((1 - Y), np.log(1 - H)).sum() #etc

    #Coste sin regularizar
    return -1 / m * (costeUno + costeCero)


def reg_network_cost(H, Y, lamda, theta1, theta2):
    '''
    Calcula el coste (regularizado) para la red neuronal,
    con una salida de la red H y la Y de los ejemplos de entrenamiento
    '''
    #Variables auxiliares
    m = Y.shape[0]

    #Coste sin regularizar
    cost = network_cost(H, Y)

    #Término de regularización (las columnas de 1's de thetas las quitamos)
    thetaSum = ((theta1[:, 1:]**2).sum() + (theta2[:, 1:]**2).sum())
    regTerm = lamda / (2 * m) * thetaSum

    #Coste regularizado
    return (cost + regTerm)

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

#TODO: hacer el backprop y el gradiente
def back_prop (params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    '''
    Propagación inversa en la red neuronal para hallar el coste y el gradiente
    '''
    #theta1 = np.reshape (params_rn[:num_ocultas ∗ (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    #theta2 = np.reshape (params_rn[num_ocultas ∗ (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

    for t in range(m):
        a1t = a1[t, :] # (1, 401)
        a2t = a2[t, :] # (1, 26)
        ht = h[t, :] # (1, 10)
        yt = y[t] # (1, 10)
        d3t = ht - yt # (1, 10)
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t)) # (1, 26)
        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

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
    y = data['y'].ravel() #(5000,)

    # Atributos
    m = len(y)
    input_size = X.shape[1]
    num_etiquetas = 10

    #Porque están de 1 - 10 y los queremos del 0 - 9
    y = (y - 1) 
    y_onehot = np.zeros((m, num_etiquetas)) #5000 x 10
    
    #Inicializamos y_onehot
    for i in range(m):
        y_onehot[i][y[i]] = 1

    # Visualizar los 100 ejemplos
    sample = np.random.choice(X.shape[0], 100)
    fig, ax = displayData(X[sample])
    #plt.show()

    # Guardamos las matrices de theta (leídas de archivo) en una lista
    weights = loadmat ( "ex4weights.mat" )
    theta1, theta2 = weights ["Theta1"], weights ["Theta2"]
    # Theta1 es de dimensión 25 x 401
    # Theta2 es de dimensión 10 x 26
    thetas = [theta1, theta2]

    # Hacemos la propagación hacia delante
    a1, z2, a2, z3, h = forward_prop(X, theta1, theta2) # z es de 5000x10

    # Calculamos el coste regularizado
    regCost = reg_network_cost(h, y_onehot, lamda, theta1, theta2)
    print("Coste regularizado:", regCost)

    # Hacemos la propagación hacia atrás


Ejercicio1(1)