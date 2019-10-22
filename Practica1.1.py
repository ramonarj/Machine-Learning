#Celia Castaños Bornaechea
#Ramón Arjona Quiñones

#TODO: 
# 1. Hacer que se dibuje la gráfica 3D

import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def carga_csv(file_name):
    """carga el fichero csv especificado y lo devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).values
    
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)


def h(x, theta):
    return np.dot(x, theta[np.newaxis].T)

def coste(X, Y, Theta):
    #Calculamos la función de coste
    m = X.shape[0]
    sum3 = 0
    for k in range(0, m):
        sum3 += (h(X[k], Theta) - Y[k])**2
    return 1 / (2*m) * sum3

def descenso_gradiente(X, Y, alpha, num_iter):

    m = X.shape[0]
    n = X.shape[1]
    unos = np.ones((m, 1))
    X = np.hstack((unos, X))
     #2.Calculamos la recta
    theta = np.array([0,0], dtype=float)
    #Bucle
    htheta = np.zeros((m, 1))

    costes = np.ones(num_iter)
    #Cada iteracíón del proceso
    for i in range (num_iter):
        #Calcular el valor de h(theta) con la theta de la iteración anterior
        htheta = h(X, theta)

        #Calcular la nueva theta (0 y 1)
        #Sumatorio de theta0
        for k in range(0, n + 1):
            sum1 = 0
            for l in range(0, m):
                sum1 += (htheta[l] - Y[l]) * X[l, k]
            theta[k] = theta[k] - alpha * (1 / m) * sum1

        #Calculamos la función de coste
        sum3 = 0
        for k in range(0, m):
            sum3 += (htheta[k] - Y[k])**2
        costes[i] = 1 / (2*m) * sum3

    return [theta, costes]


#Dibujamos la gráfica con la recta de regresión
def drawLinearRegression(X, Y, f):
    plt.figure()
    plt.plot(X, Y, 'x',color="red")
    plt.plot(X, f, color="blue")
    plt.title("Practica 1")
    plt.xlabel("Población de la ciudad en 10000s")
    plt.ylabel("Ingresos en $10000s")
    plt.legend()
    plt.savefig('time.png')
    plt.show()

    #Dibujamos la gráfica con la recta de regresión
def drawCostSurface(Theta0, Theta1, Coste):
    #Dibujamos la gráfica de coste con las thetas
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    #Plot de surface
    surf = ax.plot_surface(Theta0, Theta1, Coste, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
    ax = fig.gca(projection = '3d')
    ax.set_xlabel("θ0")
    ax.set_ylabel("θ1")
    ax.set_zlabel("J(θ)")
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    plt.show()

def make_data(t0_range, t1_range, X, Y):
    #Obtenemos las thetas
    step = 0.1
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)

    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)

    m = X.shape[0]
    unos = np.ones((m, 1))
    X = np.hstack((unos, X))

    #Obtenemos los costes
    Coste = np.empty_like(Theta0)
    for ix, iy in np.ndindex(Theta0.shape):
        Thetai = np.array([Theta0[ix,iy], Theta1[ix, iy]])
        Coste[ix, iy] = coste(X, Y, Thetai)

    return [Theta0, Theta1, Coste]

def main():
    #1.Importamos los datos a una matriz y rellenamos con 1's la primera columna para el p.escalar 
    valores = carga_csv("ex1data1.csv")
    alpha = 0.01
    X = valores[:, :-1]
    Y = valores[:, -1]

    m = X.shape[0]
    unos = np.ones((m, 1))

    theta, costes = descenso_gradiente(X, Y, alpha, 1500)
    f = h(np.hstack((unos, X)), theta)

    drawLinearRegression(X, Y, f)

    #Theta0, Theta1, coste = make_data([-10, 10], [-1, 4], X, Y)
    #drawCostSurface(Theta0, Theta1, coste)
    

main()