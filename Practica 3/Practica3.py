## Ramón Arjona Quiñones
## Celia Castaños Bornaechea

import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.integrate 
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures 
from scipy.io import loadmat

## Calcula el sigmoide del número z
def sigmoid(z): #Ej. 1.2
    s = 1 / (1 + np.exp(-z))
    return s

## Calcula el coste sobre los ejemplos de entrenamiento para un cierto valor de theta
def cost(theta, X, Y):
    H = sigmoid(np.matmul(X, theta))
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))
    return cost

## Calcula el coste de forma regularizada para un cierto lambda
def regularizedCost(theta, lamda, X, Y):
    regCost = cost(theta, X, Y)

    #Añadimos el sumatorio de thetas al cuadrado al valor del coste normal
    regCost = regCost + (lamda / 2*(len(X))) * np.sum(theta**2) 
    return regCost

## Calcula el gradiente sobre los ejemplos de entrenamiento para un cierto valor de theta
def gradiente(theta, X, Y):
    H = sigmoid(np.matmul(X, theta))
    grad = (1/ len(Y)) * np.matmul(X.T, H-Y)
    return grad

## Calcula el gradiente de forma regularizada para un cierto lambda
def regularizedGradient(theta, lamda, X, Y):
    regGrad = gradiente(theta, X, Y)
    aux = np.copy(theta)
    aux[0] = 0
    #Añadimos el sumatorio de thetas al valor del gradiente normal
    regGrad = regGrad + (lamda / len(Y) * aux)
    return regGrad

## Calcula la predicción sobre x para un cierto theta
def h(x, theta):
    return np.dot(x, theta[np.newaxis].T)

## Calcula el porcentaje de ejemplos de entrenamiento que han sido clasificados correctamente
def calcula_porcentaje(reales, predichos):
    predichos = predichos[np.newaxis].T

    # Vemos cuántos de ellos coinciden con Y y devolvemos el porcentaje sobre el total de ejemplos
    coinciden = ( reales == predichos )
    aciertos = np.sum(coinciden)

    return (aciertos / len(reales)) * 100


# Implementa la regresión logística multiclase
def oneVsAll(X, y, num_etiquetas, reg): # reg = término de regularizacion

    #Creamos la matriz de thetas
    thetas = np.zeros((num_etiquetas, X.shape[1]))

    #Clasificador para cada una de las etiquetas
    for i in range (num_etiquetas):
        # Vector de 'y' para la iteración concreta
        iterY = np.copy(y)
        if(i == 0):
            iterY = np.where (iterY == 10, 1, 0)
        else:
            iterY = np.where (iterY == i, 1, 0)

        # Calculamos el vector de pesos óptimo igual que en el ejercicio 2
        result = opt.fmin_tnc(func = regularizedCost, x0=thetas[i], fprime=regularizedGradient, args=(reg, X, iterY.ravel()))
        thetas[i] = result[0]

    return thetas


def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
    return h

## Regresión logística multiclase
def Ejercicio1(lamda):

    # Leemos los datos de las matrices (en formato .mat) con 5k ejemplos de entrenamiento
    # Cada ejemplo es una imagen de 20x20 pixeles, cada uno es un número real en escala de grises
    data = loadmat('ex3data1.mat') # Devuelve un diccionario
    X = data['X'] # Cada ejemplo de entrenamiento en una fila (5000x400)
    y = data['y'] # 1 - 10 (el 10 es 0)
    num_etiquetas = 10

    # Cogemos 10 muestras aleatorias y las pintamos
    #sample = np.random.choice(X.shape[0], 10)
    #plt.imshow(X[sample, :].reshape(-1, 20).T)
    #plt.axis('off')
    #plt.show()

    #Inicializamos theta y ponemos la columna de 1's a las X
    m = X.shape[0]
    unos = np.ones((m, 1))
    unosX = np.hstack((unos, X))

    # Resolvemos el one vs All (debería estar bien)
    thetas = oneVsAll(unosX, y, num_etiquetas, lamda)

    #Creamos la matriz
    affinities = np.zeros((num_etiquetas))
    results = np.zeros(m)

    #Calculamos las afinidades
    for i in range (m):
        for j in range (num_etiquetas):
            affinities[j] = sigmoid(np.matmul(unosX[i][np.newaxis, :], thetas[j][np.newaxis].T))
        results[i] = np.argmax(affinities)
        if(results[i] == 0):
            results[i] = 10


    print("Ha clasficado un ", calcula_porcentaje(y, results), "% de los ejemplos bien")


    
##Redes neuronales
def Ejercicio2():
    data = loadmat('ex3data1.mat') # Devuelve un diccionario
    X = data['X'] # (5000x400)
    y = data['y'] # 1 - 10 (el 10 es 0)
    num_etiquetas = 10

    m = X.shape[0]

    weights = loadmat ( "ex3weights.mat" )
    theta1, theta2 = weights ["Theta1"], weights ["Theta2"]
    # Theta1 es de dimensión 25 x 401
    # Theta2 es de dimensión 10 x 26

    h = forward_propagate(X, theta1, theta2) # h es de 5000x10

    #Creamos la matriz
    results = np.zeros(m)

    #Calculamos las afinidades
    for i in range (m):
        results[i] = np.argmax(h[i])
        if(results[i] == 0):
            results[i] = 10


    print("Ha clasficado un ", calcula_porcentaje(y, results), "% de los ejemplos bien")

    
    


#Ejercicio1(0.1)
Ejercicio2()