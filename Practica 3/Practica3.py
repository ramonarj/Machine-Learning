## Celia Castaños Bornaechea
## Ramón Arjona Quiñones

import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.integrate 
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures 

## Carga el fichero csv especificado y lo devuelve en un array de numpy
def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values
    
    # Suponemos que siempre trabajaremos con float
    return valores.astype(float)

## Redes neuronales
def Ejercicio3():
    ##TODO

Ejercicio3()