## Celia Castaños Bornaechea
## Ramón Arjona Quiñones

import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

## Carga el fichero csv especificado y lo devuelve en un array de numpy
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
    theta = np.zeros(n + 1, dtype=float)
    #Bucle
    htheta = np.zeros((m, 1))

    costes = np.ones(num_iter)
    #Cada iteracíón del proceso
    for i in range (num_iter):
        #Calcular el valor de h(theta) con la theta de la iteración anterior
        htheta = h(X, theta)

        #Calculamos el valor de las tetas
        for k in range(0, n + 1):
            sum1 = 0
            for l in range(0, m):
                sum1 += (htheta[l] - Y[l]) * X[l, k]
            theta[k] = theta[k] - alpha * (1 / m) * sum1

        #Calculamos la función de coste
        H = np.dot(X, theta)
        Aux = (H-Y)**2
        costes[i] = Aux.sum() / (2*m)

    return [theta, costes]

## Calcula el vector de pesos y coste mediante la ecuación normal
def normal(X, Y):

    m = X.shape[0]
    n = X.shape[1]
    unos = np.ones((m, 1))
    X = np.hstack((unos, X))

    #2.Calculamos la recta
    theta = np.zeros(n + 1, dtype=float)

    #Calculamos theta 
    aux = np.linalg.pinv(np.matmul(X.T, X))
    theta = np.matmul(np.matmul(aux, X.T), Y)

    #Calculamos la función de coste
    H = np.dot(X, theta)
    Aux = (H-Y)**2
    coste = Aux.sum() / (2*m)

    #Celia no tiene tetas en esa foto
    return [theta, coste]

## Dibuja la gráfica con la recta de regresión
def drawLinearRegression(X, Y, f, Xlabel, Ylabel):
    plt.figure()
    plt.plot(X, Y, 'x',color="red")
    plt.plot(X, f, color="blue")
    plt.title("Regresión lineal con descenso de gradiente")
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)


## Dibuja la superficie de coste
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
    plt.title("Coste de la regresión lineal")
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    ax.view_init(elev=10, azim=-135) #Para rotar la figura en el eje Z 

## Dibuja el contorno del coste
def drawContour(Theta0, Theta1, Coste):
    plt.figure()
    plt.xlabel("θ0")
    plt.ylabel("θ1")
    plt.title("Coste de la regresión lineal")
    plt.contour(Theta0, Theta1, Coste, np.logspace(-2, 3, 20))

## Dibuja la gráfica de relación alpha/coste
def drawLearningRate(alpha, costes, kolor):
    size = costes.shape[0]
    X = np.linspace(0, size, size)
    plt.plot(X, costes, color=kolor, label=alpha)

## Dibuja varias learningRate
def drawAlphasCost(alphas, X, Y, num_iter, colors):
    plt.figure()
    for i in range (len(alphas)):
        thetas, costes = descenso_gradiente(X, Y, alphas[i], num_iter)
        drawLearningRate(alphas[i], costes, colors[i])
    plt.title("Comparativa de costes (con gradiente)")
    plt.xlabel("Nº iteraciones")
    plt.ylabel("J(θ)")
    plt.legend(loc='upper right', title="Valores de alpha")
    
## Crea vectores de pesos con valores dentro en un intervalo dado y calcula el coste para cada uno
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

## Normaliza los valores d la X (si recibe mu y sigma, lo normaliza respecto a esos)
def normaliza(X, mu=None, sigma=None):
    # Nos pasan mu y sigma existentes
    if(mu is None or sigma is None):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)

    # Calculamos el valor noramlizado
    X = (X - mu) / sigma
    return X, mu, sigma


## Regresión lineal con una variable mediante descenso de gradiente
def Ejercicio1(alpha, num_iter):
    #1.Importamos los datos a una matriz y rellenamos con 1's la primera columna para el p.escalar 
    valores = carga_csv("ex1data1.csv")
    X = valores[:, :-1]
    Y = valores[:, -1]

    m = X.shape[0]
    unos = np.ones((m, 1))

    # Hacemos el descenso de gradiente
    theta, costes = descenso_gradiente(X, Y, alpha, num_iter)
    f = h(np.hstack((unos, X)), theta)

    ## Gráficas:
    # 1. Gráfica de la regresión lineal 
    drawLinearRegression(X, Y, f, "Población de la ciudad en 10000s", "Ingresos en $10000s")
    plt.savefig('linearRegression.pdf')
    plt.close()

    # 2. Gráfica para observar la relación entre el vector de pesos y el coste de la regresión
    Theta0, Theta1, coste = make_data([-10, 10], [-1, 4], X, Y)
    drawCostSurface(Theta0, Theta1, coste)
    plt.savefig('costSurface.pdf')
    plt.close()

    # 3. Contorno de la gráfica anterior
    drawContour(Theta0, Theta1, coste)
    plt.savefig('costContour.pdf')
    plt.close()

# Regresión lineal multivariable mediante normalización
def Ejercicio2(alpha, num_iter):
    ## 1.Importamos los datos a una matriz y los separamos en X e Y
    valores = carga_csv("ex1data2.csv")
    X = valores[:, :-1]
    Y = valores[:, -1]
    m = X.shape[0]
    unos = np.ones((1,))

    # Valores para los alphas y los colores que usaremos
    alphas = [0.3, 0.1, 0.03, 0.01, 0.005]
    colors = ["red", "orange", "yellow", "green", "cyan", "blue"]

    # Normalizamos los elementos de la muestra
    Xnorm, muX, sigmaX = normaliza(X)
    
    # Calculamos theta y el coste con ambas formas
    theta, coste = normal(X, Y) #ec.normal
    thetas, costes = descenso_gradiente(Xnorm, Y, alpha, num_iter) #gradiente
    
    # Dibujamos la comparativa de costes con los distintos valores de alpha
    drawAlphasCost(alphas, Xnorm, Y, num_iter, colors)
    plt.savefig("gradientCost_" + str(num_iter) + "iter.pdf")
    plt.close()


    ### 2. Ejemplo para comparar el modelo del gradiente con el normalizado
    ejemplo = np.array([1650, 3])
    ejemploNorm, mu, sigma = normaliza(ejemplo, muX, sigmaX)

    # Hacemos una comparativa de la forma normalizada vs el descenso de gradiente
    prediccionNormal = h(np.hstack((unos, ejemplo)), theta)[0]
    prediccionGradiente = h(np.hstack((unos, ejemploNorm)), thetas)[0]
    dif = round((1 - abs(prediccionNormal / prediccionGradiente)) * 100, 4) #Porcentaje de diferencia

    # Dibujamos la comparativa con el ejemplo
    ejemplo = ejemplo[0] / ejemplo[1] #Para unificar los 2 atributos y poder pintarlo en una gráfica normal
    plt.figure()
    plt.plot(ejemplo, round(prediccionGradiente / 100000, 3), 'x',color="green", label="Gradiente")
    plt.plot(ejemplo, round(prediccionNormal / 100000, 3), 'x',color="red", label="Normal")
    plt.legend(loc="upper right", title="Predicciones")
    plt.title("Comparativa de la predicción")
    plt.xlabel("Tamaño medio de habitación (pies cuadrados)")
    plt.ylabel("Coste en 100000$")
    plt.title(str(dif) + "% de diferencia")

    plt.savefig('normalizeExample.pdf')
    plt.close()

Ejercicio1(0.01, 1500)
Ejercicio2(0.01, 600)