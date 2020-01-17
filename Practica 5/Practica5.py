## Celia Castaños Bornaechea
## Ramón Arjona Quiñones

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.optimize as opt
import scipy.io as io

from sklearn.preprocessing import PolynomialFeatures

def h(x, theta):
    return np.dot(x, theta[np.newaxis].T)

def coste(X, Y, Theta):
    #Calculamos la función de coste
    m = X.shape[0]
    sum1 = 0
    for k in range(0, m):
        sum1 += (h(X[k], Theta) - Y[k])**2
    return 1 / (2*m) * sum1

def costeRegularizado(X, Y, theta, lamda):
    c = coste(X, Y, theta)
    n = theta.shape[0]
    m = X.shape[0]

    sum2 = 0
    for k in range(1, n):
        sum2 += theta[k]**2
    
    aux = (lamda / (2*m))*sum2
    return c + aux

# def gradiente():
#         # Initialize grad.
#     grad = np.zeros(theta.shape)

#     # Compute gradient for j >= 1.
#     grad = (1 / m) * np.dot(X.T, np.dot(X, theta) - y) + (lambda_coef / m ) * theta
    
#     # Compute gradient for j = 0,
#     # and replace the gradient of theta_0 in grad.
#     unreg_grad = (1 / m) * np.dot(X.T, np.dot(X, theta) - y)
#     grad[0] = unreg_grad[0]

#     return (cost, grad.flatten())

## Ejercicio 2: regresión logística y regularización (recibe el valor de lambda)
def Ejercicio2():
    #Leemos los valores de la matriz de datos y los guardamos
    datos = io.loadmat("ex5data1.mat")
    trainingX, trainingY = datos ["X"], datos ["y"]
    validationX, validationY = datos ["Xval"], datos ["yval"]
    testX, testY = datos ["Xtest"], datos ["ytest"]

    #Añadimos la columna de unos
    m = trainingX.shape[0]
    unos = np.ones((m, 1))
    unosX = np.hstack((unos, trainingX))

    theta = np.array([[1], [1]])
    print(costeRegularizado(unosX, trainingY, theta, 1))
   


Ejercicio2()
