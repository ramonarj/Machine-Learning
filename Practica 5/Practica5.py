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

def gradienteRegularizado(X, Y, theta, lamda):
    n = theta.shape
    gradReg = np.zeros(n)
    m = X.shape[0]
    
    sum0 = 0
    for i in range(0, m):
        sum0 += (h(X[i], theta) - Y[i])*(X[i, 0])
    gradReg[0] = (1/m)*(sum0)

    sum1 = 0
    for i in range(0, m):
        sum1 += (h(X[i], theta) - Y[i])*(X[i, 1])
    gradReg[1] = ((1/m)*(sum1)) + ((lamda/m)*theta[1])
    
    return gradReg

def costeYGradiente(X, Y, theta, lamda):
    g = gradienteRegularizado(X, Y, theta, lamda)
    c = costeRegularizado(X, Y, theta, lamda)
    return c, g

def entrenamientoMinimizarTheta(X, Y, lamda):
    
    theta = np.zeros(X.shape[1])
    
    def costFunction(theta):
        return costeYGradiente(X, Y, theta, lamda)
    
    theta = opt.minimize(fun=costFunction, x0=theta, method='CG', jac=True, options={'maxiter':200})
    return theta.x

def pintaRecta(X, Y, theta):
    plt.figure(figsize=(8, 6))
    plt.xlabel('Cambios nivel agua (X)')
    plt.ylabel('Derrame de agua (Y)')
    plt.title('Figure 2: Recta ajustada')
    plt.plot(X, Y, 'rx')
    plt.plot(X, np.dot(np.insert(X, 0, 1, axis=1), theta))
    plt.show()


def Ejercicio1():
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
    print(gradienteRegularizado(unosX, trainingY, theta, 1))

    
    theta = entrenamientoMinimizarTheta(unosX, trainingY, 0)
    print(theta)

    pintaRecta(trainingX, trainingY, theta)

Ejercicio1()
