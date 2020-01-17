## Ramón Arjona Quiñones
## Celia Castaños Bornaechea

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures 
from scipy.io import loadmat
from ML_utilities import h, hMatrix, sigmoid, sigmoidGradient, regularizedCost, regularizedGradient
from displayData import displayData
from checkNNGradients import checkNNGradients

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
def forward_prop(X, theta1, theta2):
    '''
    Propagación hacia delante en la red neuronal
    '''
    m = X.shape[0]

    a1 = X
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)

    return a1, z2, a2, z3, h

#TODO: hacer el backprop y el gradiente
def back_prop (nn_params, num_entradas, num_ocultas, num_etiquetas, X, y, lamda):
    """
    Implementa la propagación hacia atrás de la red neuronal con 2 capas
    Tenemos que convertir el vector "nn_params" en 2 matrices, ya que viene 
    desenrollado. 

    Devuelve el coste y el vector de gradientes (desenrollado también)
    """
    # 1. Volvemos a construir las matrices de pesos
    theta1 = np.reshape(nn_params[:num_ocultas * (num_entradas + 1)],
                        (num_ocultas, num_entradas + 1)) # (25,401)
    theta2 = np.reshape(nn_params[num_ocultas * (num_entradas + 1):],
                        (num_etiquetas, num_ocultas + 1)) # (10,26)

    # Número de ejemplos de entrenamiento
    m = X.shape[0]
    X = np.hstack([np.ones([m, 1]), X]) #Para el término indep.

    # 2. Hacemos la propagación hacia delante
    a1, z2, a2, z3, h = forward_prop(X, theta1, theta2) # z es de 5000x10

    # 4. Inicializamos las matrices delta
    delta1 = np.zeros((num_ocultas, num_entradas + 1)) # (25,401)
    delta2 = np.zeros((num_etiquetas, num_ocultas + 1)) # (10,26)
    
    # 5. RETRO - PROPAGACIÓN
    for t in range(m):
        a1t = a1[t, :] # (1, 401)
        a2t = a2[t, :] # (1, 26)
        ht = h[t, :] # (1, 10)
        yt = y[t] # (1, 10)

        #Error en la capa de salida
        d3t = ht - yt # (1, 10)
        #Error en la capa oculta
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t)) # (1, 26)

        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    # 6. Calculamos el coste regularizado
    regCost = reg_network_cost(h, y, lamda, theta1, theta2)
    
    # Calculamos el gradiente dividiendo lo calculado en el bucle entre los "m" ejemplos de entrenamiento
    delta1 = delta1 / m
    delta2 = delta2 / m

    #Regularizamos el gradiente
    delta1[:,1:] = delta1[:,1:] + (lamda / m) * theta1[:,1:]
    delta2[:,1:] = delta2[:,1:] + (lamda / m) * theta2[:,1:]
    
    # Desenrollamos el gradiente y lo devolvemos
    grad = np.concatenate((delta1.ravel(), delta2.ravel()))

    return regCost, grad


def pesosAleatorios(L_in, L_out):
    """
    Inicializa una matriz de pesos con valores aleatorios dentro de un rango epsilon
    """
    # Rango
    epsilon = 0.12
    
    # Inicializamos la matriz con 0s
    pesos = np.zeros((L_out, 1 + L_in))
    
    # Valores aleatorios en ese intervalo
    pesos = np.random.rand(L_out, 1 + L_in) * (2 * epsilon) - epsilon
    return pesos

def Ejercicio1(lamda, num_iter):
    '''
    Redes neuronales
    '''
    #Cargamos los datos
    data = loadmat('ex4data1.mat') 
    X = data['X'] # (5000x400)
    y = data['y'].ravel() #(5000,)
    y = (y - 1) #Porque están de 1 - 10 y los queremos del 0 - 9
    m = X.shape[0]

    # Atributos de la red neuronal
    num_entradas = 400
    num_ocultas = 25
    num_etiquetas = 10

    # 3. Inicializamos y_onehot
    y_onehot = np.zeros((m, num_etiquetas)) #5000 x 10
    for i in range(m):
        y_onehot[i][y[i]] = 1

    # Visualizar 100 ejemplos
    #sample = np.random.choice(X.shape[0], 100)
    #fig, ax = displayData(X[sample])
    #plt.show()

    # Pesos óptimos (para comprobar el coste)
    #weights = loadmat ( "ex4weights.mat" )
    #theta1, theta2 = weights ["Theta1"], weights ["Theta2"] #25x401 y 10x26

    # Damos unos pesos aleatorios para luego entrenarlos
    theta1 = pesosAleatorios(num_entradas, num_ocultas)
    theta2 = pesosAleatorios(num_ocultas, num_etiquetas)
    
    # Unimos los 2 en un solo vector
    nn_params = np.concatenate((theta1.ravel(), theta2.ravel()))

    # Hacemos la propagación hacia atrás, obteniendo el coste y el gradiente
    #regCost, grad = back_prop(nn_params, num_entradas, num_ocultas, num_etiquetas, X, y_onehot, lamda)
    # Chequeamos que el gradiente está bien
    #diff = checkNNGradients(back_prop, lamda)
    #print("Coste regularizado:", regCost)
    #print("Gradiente:", grad)

    # Llamamos a la función minimize para obtener las matrices de pesos óptimas
    #(las que hacen que haya un mínimo en el coste devuelto)
    thetaOpt = minimize(fun=back_prop,
                       x0=nn_params,
                       args=(num_entradas,
                             num_ocultas,
                             num_etiquetas,
                             X, y_onehot, lamda),
                       method='TNC',
                       jac=True,
                       options={'maxiter':num_iter})

    thetaOpt = thetaOpt.x

    # Tenemos que reconstruir los pesos a partir del vector
    theta1 = np.reshape(thetaOpt[:num_ocultas * (num_entradas + 1)],
                        (num_ocultas, num_entradas + 1)) 
    theta2 = np.reshape(thetaOpt[num_ocultas * (num_entradas + 1):],
                        (num_etiquetas, num_ocultas + 1)) 

    # Con los pesos óptimos obtenidos, hacemos la propagación para ver la efectividad de la red
    unosX = np.hstack([np.ones([m, 1]), X])
    a1, z2, a2, z3, h = forward_prop(unosX, theta1, theta2) 

    porcentaje = calcula_porcentaje(y, h, 3)
    print("La red clasificado bien un",  porcentaje, " % de los ejemplos")


Ejercicio1(1, 70)