## Celia Castaños Bornaechea
## Ramón Arjona Quiñones

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.optimize as opt
import scipy.io as io
# from ML_utilities import h, hMatrix, sigmoid, regularizedCost, regularizedGradient

from sklearn.preprocessing import PolynomialFeatures

def h(x, theta):
    return np.dot(x, theta[np.newaxis].T)

def coste(X, Y, theta):
    #Calculamos la función de coste
    m = X.shape[0]
    sumatorio =  np.sum(np.square(np.dot(X, theta) - Y))
    return (1 / (2 * m)) * sumatorio

def costeRegularizado(X, Y, theta, lamda):

    m = X.shape[0]

    theta = theta.reshape(-1, Y.shape[1])
    term0 = coste(X, Y, theta)
    sumatorio = np.sum(np.square(theta[1:len(theta)]))
    term1 = (lamda / (2 * m)) * sumatorio
    cost = term0 + term1
    return cost

def gradiente(X, Y, theta):
    m = X.shape[0]

    grad = (1 / m) * np.dot(X.T, np.dot(X, theta) - Y)
    return grad

def gradienteRegularizado(X, Y, theta, lamda):

    m = X.shape[0]
    theta = theta.reshape(-1, Y.shape[1])
    # n = theta.shape[0]
    # # gradReg = np.zeros(n)

    auxGrad = gradiente(X, Y, theta)
    gradReg = auxGrad + (lamda / m ) * theta
    
    gradReg[0] = auxGrad[0]
    return gradReg.flatten()

def costeYGradiente(X, Y, theta, lamda):
    c = costeRegularizado( X, Y, theta, lamda)
    g = gradienteRegularizado(X, Y, theta, lamda)
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

# Create a function that generates the errors.
def curvaAprendizaje(X, Y, validationX, validationY, lamda):
    m = X.shape[0]
    
    errorEntrenamiento = np.zeros(m)
    errorValidation   = np.zeros(m)
    
    
    for i in range(0, m):
        j = i + 1
        theta = entrenamientoMinimizarTheta(X[:j], Y[:j], lamda)
    
        errorEntrenamiento[i] = costeYGradiente(X[:j], Y[:j], theta, lamda)[0]
        errorValidation[i] = costeYGradiente(validationX, validationY, theta, lamda)[0]
        
    return errorEntrenamiento, errorValidation

def generaDatos(X, grado):

    Xres = X
    for i in range(1, grado):
        auxX = np.power(X, i+1)
        Xres = np.column_stack((Xres, auxX))   
    return Xres

    # print (Xres)
def normaliza(X, mu=None, sigma=None):
    # Nos pasan mu y sigma existentes
    if(mu is None or sigma is None):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)

    # Calculamos el valor normalizado
    X = (X - mu) / sigma
    return X, mu, sigma

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

def Ejercicio2():
    datos = io.loadmat("ex5data1.mat")
    trainingX, trainingY = datos ["X"], datos ["y"]
    validationX, validationY = datos ["Xval"], datos ["yval"]
    testX, testY = datos ["Xtest"], datos ["ytest"]


    m = trainingX.shape[0]
    unosEntreno = np.ones((m, 1))
    entrenoUnosX = np.hstack((unosEntreno, trainingX))

    n = validationX.shape[0]
    unosValidation = np.ones((n, 1))
    validationUnosX = np.hstack((unosValidation, validationX))
    

    errorEntrenamiento, errorValidation = curvaAprendizaje(entrenoUnosX, trainingY, validationUnosX, validationY, 0)

    print('# Training Examples\tTrain Error\tCross Validation Error\n')
    for i in range(m):
        print('{}\t\t\t{:f}\t{:f}\n'.format(i+1, float(errorEntrenamiento[i]), float(errorValidation[i])))

    plt.figure(figsize=(8, 6))
    plt.xlabel('Ejemplos de entrenamiento')
    plt.ylabel('Errores')
    plt.title('Figura 3: Curva de aprendizaje de la regresión lineal')
    plt.plot(range(0,m), errorEntrenamiento, 'b', label='Entrenamiento')
    plt.plot(range(0,m), errorValidation, 'y', label='Validación')
    plt.legend()
    plt.show()

def plotFit(min_x, max_x, mu, sigma, theta, p):

    x = np.array(np.arange(min_x - 15, max_x + 25, 0.05))

    # Map the X values.
    X_poly = generaDatos(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma

    # Add ones.
    X_poly = np.insert(X_poly, 0, 1, axis=1)

    # Plot
    plt.plot(x, np.dot(X_poly, theta), '--')

def Ejercicio3():
    datos = io.loadmat("ex5data1.mat")
    trainingX, trainingY = datos ["X"], datos ["y"]
    validationX, validationY = datos ["Xval"], datos ["yval"]
    testX, testY = datos ["Xtest"], datos ["ytest"]

    print(trainingX)
    newX = generaDatos(trainingX, 8)
    normX, muX, sigmaX = normaliza(newX)

    m = normX.shape[0]
    unosEntreno = np.ones((m, 1))
    entrenoUnosX = np.hstack((unosEntreno, normX))

    theta = entrenamientoMinimizarTheta(entrenoUnosX, trainingY,0)

    plt.figure(figsize=(8, 6))
    plt.xlabel('Cambio nivel agua (X)')
    plt.ylabel('Derrame de agua (Y)')
    plt.title('Figura 4: Regrsión polinomial ($\lambda$ = 0)')
    plt.plot(trainingX, trainingY, 'rx')
    plotFit(min(trainingX), max(trainingX), muX, sigmaX, theta, 8)
    plt.show()




    errorEntreneo, errorVal = curvaAprendizaje()
    plt.figure(figsize=(8, 6))
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.title('Figure 5: Polynomial learning curve, $\lambda$ = 0')
    plt.plot(range(1,m+1), error_train, 'b', label='Train')
    plt.plot(range(1,m+1), error_val, 'g', label='Cross Validation')
    plt.legend()


# Ejercicio1()
# Ejercicio2()
Ejercicio3()
