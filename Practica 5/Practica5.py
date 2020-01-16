## Celia Castaños Bornaechea
## Ramón Arjona Quiñones

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.optimize as opt
import scipy.io as io

from sklearn.preprocessing import PolynomialFeatures


## Pinta la frontera de decisión en la regresión lineal (como una recta)
def pinta_frontera_recta(X, Y, theta):
    # Mínimo y máximo valor para cada componente de la X
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    # Hacemos un meshgrid
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))

    # Hallamos el sigmoide y cambiamos su tamaño
    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    xx1.ravel(),
    xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    # Pintamos la frontera para z = 0.5
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

## Pinta la frontera de decisión en la regresión logística (más precisa que la recta)
def plot_decisionboundary(X, Y, theta, poly):
    #Igual que pinta_frontera_recta pero con el poly_fit_transform
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))

    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')

    
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
def calcula_porcentaje(X, Y, theta):

    #Calculamos el sigmoide
    n = Y.shape[0]
    H = sigmoid(np.matmul(X, theta))

    # Ponemos los elementos de H a 1 (si hi > 0.5) o a 0 (si hi < 0.5)
    H = np.where (H > 0.5, H, 0)
    H = np.where (H < 0.5, H, 1)

    # Vemos cuantos de ellos coinciden con Y y devolvemos el porcentaje sobre el total de ejemplos
    coinciden = np.where ( Y == H )
    aciertos = len(coinciden[0])

    return (aciertos / n) * 100

## Ejercicio 2: regresión logística y regularización (recibe el valor de lambda)
def Ejercicio2():#lamda, grado):
    #Leemos los valores de la matriz de datos
    # datos = scipy.io.loadmat("ex5data1.mat")
    datos = io.loadmat("ex5data1.mat")
    trainingX, trainingY = datos ["X"], datos ["y"]
    validationX, validationY = datos ["Xval"], datos ["yval"]
    testX, testY = datos ["Xtest"], datos ["ytest"]

    #Inicializamos theta y ponemos la columna de 1's a las X
    theta = np.zeros((X.shape[1] + 1))
    m = trainingX.shape[0]
    unos = np.ones((m, 1))
    unosX = np.hstack((unos, trainingX))

    # Calculamos el vector de pesos óptimo gracias a la función fmin_tnc (que recibe el valor inicial, 
    # las funciones de coste y gradiente y los parámetros extra necesarios (X, Y y lambda))
    result = opt.fmin_tnc(func = cost, x0=theta, fprime=gradiente, args=(unosX,Y))
    theta_opt = result[0]


Ejercicio2()#10, 6)
