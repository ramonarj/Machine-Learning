## Ramón Arjona Quiñones
## Celia Castaños Bornaechea

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
from scipy.io import loadmat
from ML_utilities import h, hMatrix, sigmoid, regularizedCost, regularizedGradient, pinta_frontera_recta
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# from process_email import email2TokenList
# from get_vocab_dict import getVocabDict


def muestraDatos(X, y, title):
    """
    Muestra los datos en 2 ejes con el título dado
    """
    # Clasificamos la Y
    y = y.flatten()
    pos = y==1
    neg = y==0

    # Pintamos
    plt.title(title)
    plt.plot(X[:,0][pos], X[:,1][pos], "k+")
    plt.plot(X[:,0][neg], X[:,1][neg], "yo")


def pintaFrontera(X, y, svm, title):
    """
    Pinta una frontera de decisión (lineal / no lineal)
    """
    #Creamos el grid
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)

    #Presicción
    yp = svm.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1)
    neg = (y == 0)

    # Pintamos
    plt.figure()
    #Datos
    plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+',  label="y = 0")
    plt.scatter(X[neg, 0], X[neg, 1], color='yellow',  
    edgecolors='black', marker='o', label="y = 1")

    #Titulo y leyenda
    plt.title(title)
    plt.legend()
    #Frontera de decisión
    plt.contour(x1, x2, yp)


def calcula_porcentaje(Y, Y2, digitsNo: int):
    '''
    Calcula el porcentaje de aciertos 
    '''

    m = Y.shape[0]

    # Vemos cuántos de ellos coinciden con Y 
    coinciden = ( Y== Y2 )
    aciertos = np.sum(coinciden)

    # Porcentaje sobre el total de ejemplos redondeado
    return round((aciertos / m) * 100, digitsNo)

def Ejericio1(reg):
    '''
    Con kernel lineal
    '''
    # 1. Cargamos los datos
    data = loadmat('ex6data1.mat') 
    X = data['X'] 
    y = data['y'].ravel() 

    # Hacemos el SVM con kernel lineal (el conjunto es linalmente separable)
    svm = SVC(kernel='linear', C=reg)
    svm.fit(X, y)

    # Pintamos la frontera
    pintaFrontera(X, y, svm, "Kernel lineal C=" + str(reg))

    # Guardamos la gráfica y cerramos
    #plt.show()
    plt.savefig("SVMLineal_C=" + str(reg) + ".pdf")
    plt.close()

def Ejericio2(reg, sigma):
    '''
    Con kernel gaussiano
    '''
    # 1. Cargamos los datos
    data = loadmat('ex6data2.mat') 
    X = data['X'] 
    y = data['y'].ravel()

    # Hacemos el SVM con kernel gaussiano
    svm = SVC(kernel='rbf', C=reg, gamma=1 / (2 * sigma ** 2))
    svm.fit(X, y)

    # Pintamos la frontera
    pintaFrontera(X, y, svm, ("Kernel gaussiano C=" + str(reg) + ", σ=" + str(sigma)))

    # Guardamos la gráfica y cerramos
    #plt.show()
    plt.savefig("SVMGaussiano_C=" + str(reg) + "_sigma" + str(sigma) + ".pdf")
    plt.close()

def Ejercicio3(values):
    '''
    Con kernel gaussiano y eligiendo bien los parámetros C y σ
    '''
    # 1. Cargamos los datos
    data = loadmat('ex6data3.mat') 
    X = data['X'] 
    y = data['y'].ravel()
    Xval = data['Xval'] 
    Yval = data['yval']

    #Lista de modelos
    svm = []

    mejor = 0 #El mejor modelo con la validación
    mejor_porc = 0

    cOPt = values[0]
    sigmaOpt = values[0]

    #Todas las posibles combinaciones de modelos con los valores dados para C y sigma
    actual = 0
    for i in range (len(values)):
        for j in range (len(values)):
            # Hacemos el SVM con kernel gaussiano
            svm.append(SVC(kernel='rbf', C=values[i], gamma=1 / (2 * values[j] ** 2)))
            svm[actual].fit(X, y)

            #Vemos el porcentaje de aciertos
            h = svm[actual].predict(Xval)
            porc = calcula_porcentaje(Yval.ravel(), h, 4)

            #Si el porcentaje es mejor que el máximo, actualizamos 
            if(porc > mejor_porc):
                mejor = actual
                mejor_porc = porc
                cOpt = values[i]
                sigmaOpt = values[j]
            
            actual+=1 #Avanzamos

    print("Mejor porcentaje:" + str(mejor_porc) + "% con C=" + str(cOpt)+ ", σ=", str(sigmaOpt))


    # Pintamos la frontera
    pintaFrontera(X, y, svm[mejor], "Kernel gaussiano óptimo, C=" + str(cOpt) + ", σ=" + str(sigmaOpt) + ". Precisión:" + str(mejor_porc) + "%")
    plt.savefig("SVMGaussianoOptimo_C=" + str(cOpt) + "_sigma" + str(sigmaOpt) + ".pdf")
    plt.close()

values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
#Ejericio1(100)
#Ejericio2(1, 0.1)
Ejercicio3(values)