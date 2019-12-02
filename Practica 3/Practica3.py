## Ramón Arjona Quiñones
## Celia Castaños Bornaechea

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
from sklearn.preprocessing import PolynomialFeatures 
from scipy.io import loadmat


def h(x, theta):
    '''
    Calcula la predicción sobre x para un cierto theta
    '''
    return np.dot(x, theta[np.newaxis].T)

    
def hMatrix(x, theta):
    '''
    Calcula la predicción sobre x para una cierta matriz de thetas 
    '''
    return np.dot(x, theta.T)

def sigmoid(z): 
    '''
    Calcula el sigmoide del número z
    '''
    s = 1 / (1 + np.exp(-z))
    return s


def regularizedCost(theta, lamda: float, X, Y):
    '''
    Calcula el coste para los ejemplos, pesos y término de regularización dados
    '''
    #Variables auxiliares
    m = X.shape[0]
    H = sigmoid(np.matmul(X, theta))

    # Coste cuando Y = 1
    costeUno = np.dot(Y, np.log(H))
    # Coste cuando Y = 0
    costeCero = np.dot((1 - Y), np.log(1 - H))
    # Término de regularización (sumado al coste original)
    regTerm = lamda / (2 * m) * np.sum(theta**2) 

    return -1 / m * (costeUno + costeCero) + regTerm

def regularizedGradient(theta, lamda: float, X, Y):
    '''
    Calcula el gradiente con los ejemplos, pesos y término de regularización dados
    '''
    #Variables auxiliares
    m = X.shape[0]
    H = sigmoid(np.matmul(X, theta))

    # Cálculo del gradiente
    grad = (1/ m) * np.matmul(X.T, H-Y)

    # No queremos usar la primera componente de theta (nos la guardamos)
    aux = theta[0]
    theta[0] = 0

    # Término de regularización
    regTerm = lamda / m * theta

    #Devolvemos su valor a theta
    theta[0] = aux

    return grad + regTerm

def oneVsAll(X: np.array, y: np.array, num_etiquetas: int, reg: float):
    '''
    Implementa la regresión lineal multiclase (reg = término de regularización)
    ''' 
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

        # Calculamos el vector de pesos óptimo para ese clasificador
        thetas[i] = fmin_tnc(func = regularizedCost, x0=thetas[i], fprime=regularizedGradient, args=(reg, X, iterY.ravel()))[0]

    return thetas

def calcula_porcentaje(X, Y, thetas, digitsNo: int):
    '''
    Calcula el porcentaje de aciertos del entrenador
    '''
    # Variables auxiliares
    m = X.shape[0]
    num_etiquetas = thetas.shape[0]

    # Creamos la matriz
    affinities = np.zeros((num_etiquetas))
    results = np.zeros(m)

    # Recorremos todos los ejemplos de entrenamiento...
    for i in range (m):
        # Afinidad del ejemplo con cada clasificador
        affinities = sigmoid(hMatrix(X[i], thetas))
        # Nos quedamos con el clasificador que de el valor más alto
        results[i] = np.argmax(affinities)

    results = results[np.newaxis].T

    # Vemos cuántos de ellos coinciden con Y 
    Y = np.where (Y == 10, 0, Y) # Para que los '10' se cambien a '0'
    coinciden = ( Y == results )
    aciertos = np.sum(coinciden)

    # Porcentaje sobre el total de ejemplos redondeado
    return round((aciertos / m) * 100, digitsNo)

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
    return h

def Ejercicio1(lamda):
    '''
    Regresión logística OneVsAll
    '''
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

    print("El entrenador tiene una precisión del ", calcula_porcentaje(unosX, y, thetas, 4), "%")


def Ejercicio2():
    '''
    Redes neuronales con unos pesos ya dados
    '''
    data = loadmat('ex3data1.mat') # Devuelve un diccionario
    X = data['X'] # (5000x400)
    y = data['y'] # 1 - 10 (el 10 es 0)
    num_etiquetas = 10

    m = X.shape[0]
    unos = np.ones((m, 1))
    unosX = np.hstack((unos, X))

    weights = loadmat ( "ex3weights.mat" )
    theta1, theta2 = weights ["Theta1"], weights ["Theta2"]
    # Theta1 es de dimensión 25 x 401
    # Theta2 es de dimensión 10 x 26

    h = forward_propagate(X, theta1, theta2) # h es de 5000x10

    #print("El entrenador tiene una precisión del ", calcula_porcentaje(unosX, y, h, 4), "%")


Ejercicio1(0.1)
#Ejercicio2()