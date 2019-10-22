import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.integrate 
import scipy.optimize as opt


def carga_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).values
    
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def visualizacion_datos(X, Y): #Ejercicio 1.1
    # Obtiene un vector con los índices de los ejemplos positivos
    pos = np.where (Y == 1 )

    # Dibuja los ejemplos positivos
    plt.scatter(X[pos, 0], X[pos, 1], marker='+' ,c='k' )

    # Obtiene un vector con los índices de los ejemplos negativos
    pos = np.where (Y == 0 )

    # Dibuja los ejemplos positivos
    plt.scatter(X[pos, 0], X[pos, 1], marker='o' ,c='green' )

    #Dibujamos la gráfica con la recta de regresión
def drawLinearRegression(X, Y, f):
    visualizacion_datos(X,Y)
    plt.figure()
    plt.plot(X, Y, 'x',color="red")
    plt.plot(X, f, color="blue")

def sigmoid(z): #Ej. 1.2
    s = 1 / (1 + np.exp(-z))
    return s

def cost(theta, X, Y):
    # H = sigmoid(np.matmul(X, np.transpose(theta)))
    H = sigmoid(np.matmul(X, theta))
    # cost = (- 1 / (len(X))) * np.sum( Y * np.log(H) + (1 - Y) * np.log(1 - H) )
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))
    return cost

def gradiente(theta, X, Y):
    H = sigmoid(np.matmul(X, theta))
    grad = (1/ len(Y)) * np.matmul(X.T, H-Y)
    return grad

def h(x, theta):
    return np.dot(x, theta[np.newaxis].T)

def main():
    #Leemos los valores de la matriz, nota examen 1, nota examen 2 y si ha sido aceptado
    valores = carga_csv("ex2data1.csv")
    X = valores[:, :-1]
    Y = valores[:, -1]

    #visualizacion_datos(X, Y)

    theta = np.zeros((X.shape[1] + 1))

    m = X.shape[0]
    unos = np.ones((m, 1))
    unosX = np.hstack((unos, X))


    result = opt.fmin_tnc(func = cost, x0=theta, fprime=gradiente, args=(unosX,Y))
    theta_opt = result[0]

    print("Coste: " + str(cost(theta_opt, unosX, Y)))
    print("\nGradiente: " + str(gradiente(theta_opt, unosX, Y)))

    #f = h(unosX, theta_opt)
    #drawLinearRegression(X, Y, f)
    #plt.show()



main()