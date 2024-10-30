import numpy as np

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
    return (1 / (1 + np.exp(-z)))

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