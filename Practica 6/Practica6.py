## Ramón Arjona Quiñones
## Celia Castaños Bornaechea

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
from scipy.io import loadmat
from ML_utilities import h, hMatrix, sigmoid, regularizedCost, regularizedGradient, pinta_frontera_recta
from sklearn.svm import SVC


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
    plt.show()



# Create a funtion that plots a non-linear decision boundary.
def pintaFrontera(X, y, model, title):
    """

    """
    # Get the separating hyperplane.
    w = model.coef_[0]
    a = -w[0] / w[1]
    # Only 2 points are required to define a line, e.g. min and max.
    xx = np.array([X[:,0].min(), X[:,0].max()])
    yy = a * xx - (model.intercept_[0]) / w[1]
    # Plot the separating line.
    plt.plot(xx, yy, 'r-', linewidth=2)
    # Plot the training data.
    muestraDatos(X, y, title)

def Ejericio1(reg):
    # 1. Cargamos los datos
    data = loadmat('ex6data1.mat') 
    X = data['X'] # (5000x400)
    y = data['y'].ravel() #(5000,)

    # Hacemos el SVM con kernel lineal
    svm = SVC(kernel='linear', C=reg)
    svm.fit(X, y)

    # Pintamos la frontera
    pintaFrontera(X, y, svm, "Kernel lineal")


Ejericio1(100)