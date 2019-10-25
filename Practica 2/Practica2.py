import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.integrate 
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures 

def carga_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).values
    
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def visualizacion_datos(X, Y, labelX, labelY, title=""): #Ejercicio 1.1
    # Obtiene un vector con los índices de los ejemplos positivos
    pos = np.where (Y == 1 )

    # Dibuja los ejemplos con Y = 1
    plt.scatter(X[pos, 0], X[pos, 1], marker='+' ,c='k', label="y = 1" )

    # Obtiene un vector con los índices de los ejemplos negativos
    pos = np.where (Y == 0 )

    # Dibuja los ejemplos con Y = 0
    plt.scatter(X[pos, 0], X[pos, 1], marker='o' ,c='darkkhaki', label="y = 0"  )

    #HUD
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.title(title)
    plt.legend()
    plt.legend(loc='upper right')


def pinta_frontera_recta(X, Y, theta):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))

    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    xx1.ravel(),
    xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    # el cuarto parámetro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

    #plt.savefig("frontera.pdf")


def plot_decisionboundary(X, Y, theta, poly):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')

    

def sigmoid(z): #Ej. 1.2
    s = 1 / (1 + np.exp(-z))
    return s

def cost(theta, X, Y):
    H = sigmoid(np.matmul(X, theta))
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))
    return cost

##Estará bien??
def regularizedCost(theta, lamda, X, Y):
    regCost = cost(theta, X, Y)
    regCost = regCost + (lamda / 2*(len(X))) * np.sum(theta**2)
    return regCost

def gradiente(theta, X, Y):
    H = sigmoid(np.matmul(X, theta))
    grad = (1/ len(Y)) * np.matmul(X.T, H-Y)
    return grad

##Estará bien??
def regularizedGradient(theta, lamda, X, Y):
    regGrad = gradiente(theta, X, Y)
    aux = np.copy(theta)
    aux[0] = 0
    regGrad = regGrad + (lamda / len(Y) * aux)
    return regGrad

def h(x, theta):
    return np.dot(x, theta[np.newaxis].T)

def calcula_porcentaje(X, Y, theta):

    n = Y.shape[0]
    H = sigmoid(np.matmul(X, theta))

    H = np.where (H > 0.5, H, 0)
    H = np.where (H < 0.5, H, 1)

    coinciden = np.where ( Y == H )
    aciertos = len(coinciden[0])

    return (aciertos / n) * 100
    

def Ejercicio1():
    #Leemos los valores de la matriz, nota examen 1, nota examen 2 y si ha sido aceptado
    valores = carga_csv("ex2data1.csv")
    X = valores[:, :-1]
    Y = valores[:, -1]

    theta = np.zeros((X.shape[1] + 1))

    m = X.shape[0]
    unos = np.ones((m, 1))
    unosX = np.hstack((unos, X))


    result = opt.fmin_tnc(func = cost, x0=theta, fprime=gradiente, args=(unosX,Y))
    theta_opt = result[0]

    #Porcentaje 
    porc = calcula_porcentaje(unosX, Y, theta_opt)

    #Gráfica
    title = str(porc) + "% de ejemplos clasificados correctamente"
    visualizacion_datos(X, Y, "Exam 1 score", "Exam 2 score", title)
    pinta_frontera_recta(X, Y, theta_opt)
    plt.savefig("Ejercicio1.pdf")


def Ejercicio2(lamda):
    #Leemos los valores de la matriz de datos
    valores = carga_csv("ex2data2.csv")
    X = valores[:, :-1]
    Y = valores[:, -1]

    # Creamos el polinomio X de grado 6 a partir de combinaciones de x1 y x2
    poly = PolynomialFeatures(6)
    polyX = poly.fit_transform(X)
    theta = np.zeros((polyX.shape[1]))


    # Calculamos el vector de pesos óptimo gracias a la función fmin_tnc (que recibe el valor inicial, 
    # las funciones de coste y gradiente y los parámetros extra necesarios (X, Y y lambda))
    result = opt.fmin_tnc(func = regularizedCost, x0=theta, fprime=regularizedGradient, args=(lamda, polyX, Y))
    theta_opt = result[0]

    # Calculamos el porcentaje de evaluados correctamente
    porc = round(calcula_porcentaje(polyX, Y, theta_opt), 3)

    # Pintamos los datos y la frontera de decisión 
    title = "Lambda = " + str(lamda) + "\n" + str(porc) + "% de ejemplos clasificados correctamente"
    visualizacion_datos(X, Y, "Microchip Test 1", "Microchip Test 2", title)
    plot_decisionboundary(X, Y, theta_opt, poly)

    # Guardamos la gráfica y cerramos
    plt.savefig("Ejercicio2_lambda" + str(lamda) + ".pdf")
    plt.close()

Ejercicio2(10)