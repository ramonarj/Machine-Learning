## Ramón Arjona Quiñones
## Celia Castaños Bornaechea

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
from sklearn.preprocessing import PolynomialFeatures 
from scipy.io import loadmat
from ML_utilities import h, hMatrix, sigmoid, regularizedCost, regularizedGradient
from displayData import displayData

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
def forward_prop_generic(X, thetas, num_layers:int):
    m = X.shape[0]
    inputLayer = X

    # 1 iteración por cada capa de la matriz
    for i in range (num_layers):
        # Añadimos la columna de 1's a la entrada
        inputLayer = np.hstack([np.ones([m, 1]), inputLayer])
        # Calculamos la capa de salida
        outputLayer = hMatrix(inputLayer, thetas[i]) 
        # Capa de entrada de la siguiente iteración
        inputLayer = sigmoid(outputLayer)

    return inputLayer

#Nº nodos en cada capa: 400, 25, 10
def forward_prop(X, theta1, theta2):
    '''
    Propagación hacia delante en la red neuronal
    '''
    m = X.shape[0]

    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)

    return a1, z2, a2, z3, h

#TODO: hacer el backprop y el gradiente
def back_prop (nn_params, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    """
    Implementa la propagación hacia atrás de la red neuronal con 2 capas
    Tenemos que convertir el vector "nn_params" en 2 matrices, ya que viene 
    desenrollado. 

    Devuelve el coste y el vector de gradientes (desenrollado también)
    """
    # Reshape nn_params back into the parameters Theta1 and Theta2,
    # the weight matrices for our 2 layer neural network.
    Theta1 = np.reshape(nn_params[:num_ocultas * (num_entradas + 1)],
                        (num_ocultas, num_entradas + 1)) # (25,401)
    Theta2 = np.reshape(nn_params[num_ocultas * (num_entradas + 1):],
                        (num_etiquetas, num_ocultas + 1)) # (10,26)

    # Get the number of training examples, m.
    m = X.shape[0]
    
    # Insert a 1's column for the bias unit.
    X = np.insert(X, 0, 1, axis=1) # (5000,401)
    
    # Perform forward propagation to compute a(l) for l=1,...,L.
    # z(l+1) = theta(l)a(l) and a(l+1) = g(z(l+1)).
    z2 = np.dot(X, Theta1.T) # (5000, 25)
    a2 = sigmoid(z2) # (5000, 25)
    
    # Add 1's for the bias unit.
    a2 = np.insert(a2, 0, 1, axis=1) # (5000,26)
    z3 = np.dot(a2, Theta2.T) # (5000, 10)
    a3 = sigmoid(z3) # (5000, 10)
    
    # Create a y matrix of shape (m, K)
    # for later use in recoding.
    y_recoded = np.zeros((m, num_etiquetas)) # (5000, 10)
    
    # Initialize Delta matrices.
    D1 = np.zeros((num_ocultas, num_entradas + 1)) # (25,401)
    D2 = np.zeros((num_etiquetas, num_ocultas + 1)) # (10,26)
    
    #############################################################
    ########## Forward Propagation and Cost Computation #########
    #############################################################
    
    # Initialize cost.
    j = 0
    # Fwd pass; for training example t = 1,...,m:
    for t in range(m):
        x_t = X[t]
        
        # Recode the categorical integer values of  y
        # as vectors with all values set to zeros except
        # for one value set to "1", which indicates whether
        # it belongs to class k (yk = 1).
        y_recoded[t, y[t] - 1] = 1
            
        # Compute cost for every training example.
        j += np.sum(-y_recoded[t] * np.log(a3[t]) - (1 - y_recoded[t]) * np.log(1 - a3[t])) # float

        ###############################################################
        ########## Back Propagation and Gradients Computation #########
        ###############################################################
    
        # Compute the error delta.
        d_3 = a3[t] - y_recoded[t] # (10,)
        d_3 = d_3.reshape(-1,1) # (10,1)
        
        # Perform back propagation.
        # In the parameters Thetas_i,j, i are indexed
        # starting from 1, and j is indexed starting from  0.
        d_2 = np.dot(Theta2.T[1:,:], d_3) * sigmoid(z2[t].reshape(-1,1)) # (25,10)x(10,1).*(25,1) = (25,1)
        D1 += np.dot(d_2, x_t.reshape(-1, 1).T) # (25,401)
        D2 += np.dot(d_3, a2[t].reshape(1,-1)) # (10,26)

    # Compute total cost.
    J = j / m # float
    
    # Compute the regularization term.
    # We should not be regularizing the first column of theta,
    # which is used for the bias term.
    Theta1_sum = np.sum(np.square(Theta1[:,1:])) # float
    Theta2_sum = np.sum(np.square(Theta2[:,1:])) # float
    reg_term = Theta1_sum + Theta2_sum # float
    
    # Compute total cost with regularization.
    J = J + (reg / (2 * m)) * reg_term # float
    
    # Update our new Delta matrices with regularization.
    # We should not be regularizing the first column of theta,
    # which is used for the bias term.
    # First devide every value in Deltas with m.
    D1 = D1 / m
    D2 = D2 / m
    D1[:,1:] = D1[:,1:] + (reg / m) * Theta1[:,1:]
    D2[:,1:] = D2[:,1:] + (reg / m) * Theta2[:,1:]
    
    # Unroll gradients.
    Deltas = [D1, D2]
    unrolled_Deltas = [Deltas[i].ravel() for i,_ in enumerate(Deltas)]
    grad = np.concatenate(unrolled_Deltas)

    return J, grad


def Ejercicio1(lamda):
    '''
    Redes neuronales
    '''
    #Cargamos los datos
    data = loadmat('ex4data1.mat') 
    X = data['X'] # (5000x400)
    y = data['y'].ravel() #(5000,)

    # Atributos
    m = len(y)
    input_size = X.shape[1]
    num_etiquetas = 10

    #Porque están de 1 - 10 y los queremos del 0 - 9
    y = (y - 1) 
    y_onehot = np.zeros((m, num_etiquetas)) #5000 x 10
    
    #Inicializamos y_onehot
    for i in range(m):
        y_onehot[i][y[i]] = 1

    # Visualizar los 100 ejemplos
    sample = np.random.choice(X.shape[0], 100)
    fig, ax = displayData(X[sample])
    #plt.show()

    # Guardamos las matrices de theta (leídas de archivo) en una lista
    weights = loadmat ( "ex4weights.mat" )
    theta1, theta2 = weights ["Theta1"], weights ["Theta2"]
    # Theta1 es de dimensión 25 x 401
    # Theta2 es de dimensión 10 x 26
    thetas = [theta1.ravel(), theta2.ravel()] 

    # Unimos los 2 en un solo vector
    nn_params = np.concatenate(thetas)

    # Hacemos la propagación hacia delante
    a1, z2, a2, z3, h = forward_prop(X, theta1, theta2) # z es de 5000x10

    # Calculamos el coste regularizado
    regCost = reg_network_cost(h, y_onehot, lamda, theta1, theta2)
    print("Coste regularizado:", regCost)

    # Hacemos la propagación hacia atrás
    regCost, grad = back_prop(nn_params, 400, 25, 10, X, y, lamda)
    print("Coste regularizado:", regCost)


Ejercicio1(1)