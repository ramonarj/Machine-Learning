from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def main():

    data = loadmat ("ex3data1.mat")
    pesos = loadmat("ex3weights.mat")

    #Theta1 es de dimension 25x401
    #Theta2 es de dimension 10x26
    theta1, theta2 = pesos['Theta1'], pesos['Theta2']

    
    y = data["y"]
    for a in y:
        if(a==0):print("asdabs")
    X = data["X"]

    

    #Mostramos una seleccion aleatoria de 10 ejemplos de entrenamiento
    sample=np.random.choice(X.shape[0],10)
    plt.imshow(X[sample,:].reshape(-1,20).T)
    plt.axis('off')


    a1, z2, a2, z3, h = propagacionHaciaDelante(X,theta1,theta2)

    print(porcentajeTotal(X,h,y))


def propagacionHaciaDelante(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoide(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoide(z3)
    return a1, z2, a2, z3, h



def porcentajeTotal(X,thetaOptimo,Y):
    aciertos=0
    for i in range(X.shape[0]):
        indiceMejorTheta = np.argmax(thetaOptimo[i])
        if(indiceMejorTheta+1==Y[i]):aciertos+=1     
    return(aciertos/X.shape[0])
  

def funcionCoste(Theta,X,Y,landa):
    m=X.shape[0]
    primersumando=np.dot(np.log(sigmoide(np.dot(X,Theta))).T,Y)
    segundosumando=np.dot(np.log(1-sigmoide(np.dot(X,Theta))).T,(1-Y))
    return(-1/len(X))*(primersumando+segundosumando)+(landa/(2*m))*sumatorio(X.shape[1],Theta)

def regresionLogisticaVectorizada(Theta,X,Y,landa):
    m=X.shape[0]
    aux=np.r_[[0],Theta[1:]]
    return (1/m)*(np.dot(X.T,sigmoide(np.dot(X,Theta))-np.ravel(Y)))+(landa*aux/m)

def sigmoide(x):
    return (1/(1+np.exp(-x)))

def sumatorio(n,theta):
    result=0
    for i in range(n-1):
        result+=(theta[i+1])**2
    return result

main()