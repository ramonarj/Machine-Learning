## Ramón Arjona Quiñones
## Celia Castaños Bornaechea

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
    m = np.shape(X)[0]
    sumatorio =  np.sum(np.square(np.dot(X, theta) - Y))
    return (1 / (2 * m)) * sumatorio

def costeRegularizado(X, Y, theta, lamda):
    m = np.shape(X)[0]

    #Convierte a theta en un vector de dos dimensiones
    theta = theta.reshape(-1, Y.shape[1]) 
    #Calcula el término sin regularizar
    term0 = coste(X, Y, theta)
    #Calcula el término con regularización
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
    
    #Convierte a theta en un vector de dos dimensiones
    theta = theta.reshape(-1, Y.shape[1])
    #Calcula el término sin regularizar
    auxGrad = gradiente(X, Y, theta)
    #Calcula el término con regularización
    gradReg = auxGrad + (lamda / m ) * theta
    
    gradReg[0] = auxGrad[0]
    return gradReg.flatten()

def costeYGradiente(X, Y, theta, lamda):
    c = costeRegularizado( X, Y, theta, lamda)
    g = gradienteRegularizado(X, Y, theta, lamda)
    return c, g

def entrenamientoMinimizarTheta(X, Y, lamda):
    
    theta = np.zeros([X.shape[1], 1])
    
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

#Función que ajusta la regresión polinómica
def ajusta(minX, maxX, mu, sigma, theta, p):

    X = np.array(np.arange(minX - 15, maxX + 25, 0.05))

    #Establece las X
    Xpolinom = generaDatos(X, p)
    Xpolinom = Xpolinom - mu
    Xpolinom = Xpolinom / sigma

    Xpolinom = np.insert(Xpolinom, 0, 1, axis=1)

    plt.plot(X, np.dot(Xpolinom, theta), '-')

def curvaValidacion(X, Y, validationX, validationY):
    lambdas = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    errorEntreno = np.zeros((len(lambdas), 1))
    errorValidacion = np.zeros((len(lambdas), 1))


    print (validationY.shape[1])
    
    # Loop over lambda_vec.
    for i in range(len(lambdas)):
        lamda = lambdas[i]

        # Train the model with current lambda_coef.
        theta = entrenamientoMinimizarTheta(X, Y, lamda)

        # Get the errors with lambda_coef set to 0!!!
        errorEntreno[i] = costeYGradiente(X, Y, theta, 0)[0]
        errorValidacion[i] = costeYGradiente(validationX, validationY, theta, 0)[0]
         
    return lambdas, errorEntreno, errorValidacion

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
    #Carga de los datos
    datos = io.loadmat("ex5data1.mat")
    trainingX, trainingY = datos ["X"], datos ["y"]
    validationX, validationY = datos ["Xval"], datos ["yval"]
    testX, testY = datos ["Xtest"], datos ["ytest"]

    #Añade la columna de unos
    m = trainingX.shape[0]
    unosEntreno = np.ones((m, 1))
    entrenoUnosX = np.hstack((unosEntreno, trainingX))

    n = validationX.shape[0]
    unosValidation = np.ones((n, 1))
    validationUnosX = np.hstack((unosValidation, validationX))
    
    #Calcula la curva de aprendizaje
    errorEntrenamiento, errorValidation = curvaAprendizaje(entrenoUnosX, trainingY, validationUnosX, validationY, 0)


    plt.figure(figsize=(8, 6))
    plt.xlabel('Ejemplos de entrenamiento')
    plt.ylabel('Errores')
    plt.title('Curva de aprendizaje de la regresión lineal')
    plt.plot(range(0,m), errorEntrenamiento, 'b', label='Entrenamiento')
    plt.plot(range(0,m), errorValidation, 'y', label='Validación')
    plt.legend()
    plt.show()

def Ejercicio3():
    datos = io.loadmat("ex5data1.mat")
    trainingX, trainingY = datos ["X"], datos ["y"]
    validationX, validationY = datos ["Xval"], datos ["yval"]
    testX, testY = datos ["Xtest"], datos ["ytest"]

    #Genera nuevos datos de entrada y los normaliza
    # 1º Casos de entrenamiento
    newX = generaDatos(trainingX, 8)
    normX, mu, sigma = normaliza(newX)
    
    #2º Casos de validación
    newValX = generaDatos(validationX, 8)
    normValX = newValX - mu
    normValX = normValX / sigma

    #Añade la columna de unos
    m = normX.shape[0]
    unosEntreno = np.ones((m, 1))
    entrenoUnosX = np.hstack((unosEntreno, normX))

    n = normValX.shape[0]
    unosValidacion = np.ones((n, 1))
    validacionUnosX = np.hstack((unosValidacion, normValX))

    #Calcula el vector theta que minimiza el error con lambda = 0
    theta = entrenamientoMinimizarTheta(entrenoUnosX, trainingY,0)

    # Muestra la curva que se genera
    plt.figure(figsize=(8, 6))
    plt.xlabel('Cambio nivel agua (X)')
    plt.ylabel('Derrame de agua (Y)')
    plt.title('Figura 4: Regresión polinomial ($\lambda$ = 0)')
    plt.plot(trainingX, trainingY, 'rx')
    ajusta(min(trainingX), max(trainingX), mu, sigma, theta, 8)



    #Curvas de aprendizaje
    errorEntreno, errorValidacion = curvaAprendizaje(entrenoUnosX, trainingY, validacionUnosX, validationY, 1)
    plt.figure(figsize=(8, 6))
    plt.xlabel('Número de casos de entrenamiento')
    plt.ylabel('Error')
    plt.title('Curva de aprendizaje para regresión lineal ($\lambda$ = 1)')
    plt.plot(range(1,m+1), errorEntreno, 'b', label='Entrenamiento')
    plt.plot(range(1,m+1), errorValidacion, 'y', label='Validación')
    plt.legend()
    plt.show()

def Ejercicio4():
    datos = io.loadmat("ex5data1.mat")
    trainingX, trainingY = datos ["X"], datos ["y"]
    validationX, validationY = datos ["Xval"], datos ["yval"]
    testX, testY = datos ["Xtest"], datos ["ytest"]

    #Genera nuevos datos de entrada y los normaliza
    # 1º Casos de entrenamiento
    newX = generaDatos(trainingX, 8)
    normX, mu, sigma = normaliza(newX)
    
    #2º Casos de validación
    newValX = generaDatos(validationX, 8)
    normValX = newValX - mu
    normValX = normValX / sigma
    
    #3º Casos de test
    newTestX = generaDatos(testX, 8)
    normTestX = newTestX - mu
    normTestX = normTestX / sigma

    #Añade la columna de unos
    m = normX.shape[0]
    unosEntreno = np.ones((m, 1))
    entrenoUnosX = np.hstack((unosEntreno, normX))

    n = normValX.shape[0]
    unosValidacion = np.ones((n, 1))
    validacionUnosX = np.hstack((unosValidacion, normValX))

    o = normTestX.shape[0]
    unosTest = np.ones((o, 1))
    testUnosX = np.hstack((unosTest, normTestX))

    #Calcula el vector theta que minimiza el error con lambda = 0
    # theta = entrenamientoMinimizarTheta(entrenoUnosX, trainingY,0)
    print (trainingY.shape)
    print (validationY.shape)
    lambdas, errorEntreno, errorValidacion = curvaValidacion(entrenoUnosX, trainingY, validacionUnosX, validationY)

    
    plt.figure(figsize=(8, 6))
    plt.xlabel('$\lambda$')
    plt.ylabel('Error')
    plt.title('Figura 6: Selección del parámetro $\lambda$')
    plt.plot(lambdas, errorEntreno, 'b', label='Entrenamiento')
    plt.plot(lambdas, errorValidacion, 'y', label='Validación')
    plt.legend()
    plt.show()

    theta = entrenamientoMinimizarTheta(entrenoUnosX, trainingY, 3)

    errorTest = costeYGradiente(testUnosX, testY, theta, 0)[0]
    print("Error de los valores de Test para el mejor lambda: {0:.4f}".format(errorTest))
# Ejercicio1()
# Ejercicio2()
# Ejercicio3()
Ejercicio4()
