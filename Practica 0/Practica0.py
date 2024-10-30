## Celia Castaños Bornaechea
## Ramón Arjona Quiñones

import scipy.integrate 
import math
import numpy as np
import random 
import time
import matplotlib.pyplot as plt

#Integra iterativamente
def integra_mc_iterativo(fun, a, b, num_puntos=10000):
    tic = time.time()

    aux = np.linspace(a, b, 1000)

    #1.Hallamos el máximo de la función
    maximo = aux[0]
    for i in aux:
        if fun(i) > maximo:
            maximo = fun(i)

    count = 0

    #2.Bombardeamos con puntos y vemos cuales están por debajo de la integral
    for i in range (num_puntos):
        x = random.uniform(a, b)
        y = random.uniform(0, maximo)
        if(y <= fun(x)):
            count+=1

    #3.Calculamos el area total y la que está bajo la curva
    area = maximo * (b - a)
    sol = area * (count / num_puntos)

    toc = time.time()

    #Lo que vamos a hacer ahora ya es de dibujar
    #aux2 = np.linspace(a, b, 10000)
    #fun2 = np.vectorize(fun)
    #plt.plot(aux2, fun2(aux2), color = "red")
    #plt.figure()
    #plt.savefig('iterativo.png')

    return 1000 * (toc - tic)

#Integra usando vectores 
def integra_mc_vectores(fun, a, b, num_puntos=10000):
    tic = time.time()
    aux = np.linspace(a, b, 1000)

    #Hallamos el máximo de la función
    fun2 = np.vectorize(fun)
    maximo = np.amax(fun2(aux))

    #Calculamos puntos aleatorios
    x = np.random.uniform(a, b, num_puntos)
    y = np.random.uniform(0, maximo, num_puntos)

    #Vectorizamos la función
    bajo_curva = np.where(y < fun2(x))
    count = len(bajo_curva[0])

    #3.Calculamos el area total y la que está bajo la curva
    area = maximo * (b - a)
    sol = area * (count / num_puntos)

    toc = time.time()
    return 1000 * (toc - tic)

#Main
def main():

    tiempos_iterativos = []
    tiempos_vectores = []
    sizes = np.linspace(100, 100000, 20)
    for size in sizes:
        tiempos_iterativos += [integra_mc_iterativo(math.sin, 0, math.pi/2, int(size))]
        tiempos_vectores += [integra_mc_vectores(math.sin, 0, math.pi/2, int(size))]

    plt.figure()
    plt.scatter(sizes, tiempos_iterativos, c="red", label="iterativo")
    plt.scatter(sizes, tiempos_vectores, c="blue", label="vectores")
    plt.title("Integración por Montecarlo")
    plt.xlabel("Nº puntos")
    plt.ylabel("Milisegundos")
    plt.legend()
    plt.savefig('time.png')
    plt.show()

#Llamamos al main
main()