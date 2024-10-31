## Ramón Arjona Quiñones
## Celia Castaños Bornaechea

# numpy # (para todo)
import numpy as np
# matplotlib # (para dibujar)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# sklearn # (para SVM y para dividir los conjuntos de datos)
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
# scipy # (para cargar y guardar matrices)
from scipy.io import savemat, loadmat
# opencv # (para leer imágenes)
import cv2
# sistema operativo # (para los paths)
import os
# tqdm # (para la barra de progreso)
from tqdm import tqdm
from tqdm import trange
# nuestro #
from ML_utilities import trainNeutralNetwork, forward_prop, calcula_porcentaje, calcula_porcentaje_Y, sigmoid, hMatrix, oneVsAll, makeOneHot, getBestSVMMultiClass
# joblib # (para guardar y cargar las SVM)
from joblib import dump, load

# RECURSOS
IMG_SIZE = 64 # Con esto parece que es suficiente (no hay diferencia con subirlo a 32 o 64, y como tenemos muchos datos, nos vale)
RES_PATH = 'flowers/'
FLOWER_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
FLOWER_COUNT = [0, 0, 0, 0, 0] #Se rellena solo
SAMPLE_COUNT = 200 #Número de flores de cada tipo

# DATOS CRUDOS (SIN DIVIDIR)
X = []
y = []

# PORCENTAJES PARA DIVIDIR EL DATASET (60-20-20)
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2 
TEST_FRACTION = 0.2 

# Para cargar y guardar matrices de datos
DATA_FILENAME = "flowersData.mat"
WEIGHTS_NAMES = ["OneVsAllWeights.mat", "NetworkWeights.mat", "SVM.joblib"]

def LoadAllImages():
    '''
    Carga todas las imágenes
    '''
    print("··· Cargando las imágeners de las flores ··· ")
    for i in range (len(FLOWER_NAMES)):
        LoadFlowerImages(i)

    # Devolvemos los sets ya divididos
    return DivideSets(X, y)

def LoadFlowerImages(flowerType):
    '''
    Carga las imágenes del directorio especificado y rellena la Y (one_hot) con el índice especificado
    '''
    folder = RES_PATH + FLOWER_NAMES[flowerType]
    indice = 0
    #Calculamos el índice del 1er elemento que vamos a meter
    for i in range(flowerType):
        indice+=FLOWER_COUNT[i]

    #Recorre las imágenes de ese directorio
    images = os.listdir(folder)[:SAMPLE_COUNT]
    for img in tqdm(images): 
        # label = assign_label(img,flower_type)
        path = os.path.join(folder,img) #Path de la imagen
        img = cv2.imread(path, cv2.IMREAD_COLOR) #Lee la imagen
        new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) #La reescala a SIZEXSIZE

        #Añade la imagen a los casos de entrenamiento
        X.append(np.array(new_array / 255).ravel()) 
        y.append(flowerType)
        FLOWER_COUNT[flowerType] += 1 #Añadimos una flor al recuento


def DivideSets(X, y):
    '''
    Divide el dataset en 3 sets de entrenamiento, validación y test
    '''
    #Dividimos en entrenamiento / test (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(y), test_size=TEST_FRACTION, random_state=42)
    #Volvemos a dividir en entrenamiento / validación, para tener un total de 60% (train), 20% (val), 20% (test)
    X_train, X_val, y_train, y_val = train_test_split(np.asarray(X_train), np.asarray(y_train), test_size=0.25, random_state=42)
    #X_mock, X_val, y_mock, y_val = train_test_split(np.asarray(X_train), np.asarray(y_train), test_size=0.25, random_state=42)

    #Devolvemos una tupla con los conjuntos troceados
    return [np.asarray(X_train), np.asarray(y_train), np.asarray(X_val), np.asarray(y_val), np.asarray(X_test), np.asarray(y_test)]
        

# Regresión logística multiclase #
def LogisticRegressionClassifier(data:dict, lamdas:tuple):
    '''
    Clasificador por regresión logística; entrena el modelo
    buscando el lamda óptimo y guarda los pesos en un archivo .mat
    '''
    # Atributos
    num_etiquetas = len(FLOWER_NAMES)
    
    # Sacamos los conjuntos del diccionario
    X_train = data["X_train"]
    y_train = data["y_train"].ravel()
    X_val = data["X_val"]
    y_val = data["y_val"].ravel()

    # Resolvemos el one vs All
    bestLamda = lamdas[0]
    bestPorc = 0
    bestThetas = np.zeros((num_etiquetas, X_train.shape[1] + 1))

    # Encontramos el mejor valor de lamda usando el conjunto de validación
    print("··· Haciendo el descenso de gradiente ··· ")
    for i in range(len(lamdas)):
        print(" Probando con lamda = " + str(lamdas[i]))
        # 1. Primero entrenamos el modelo
        thetas = oneVsAll(X_train, y_train, num_etiquetas, lamdas[i])

        # 2. Luego vemos su precisión sobre el conjunto de validación
        m = X_val.shape[0]
        unosX = np.hstack([np.ones([m, 1]), X_val])
        z = sigmoid(hMatrix(unosX, thetas))
        porc = calcula_porcentaje(y_val, z, 4)

        print("* " + str(porc) + "% *")

        # 3. Actualizamos si es mejor con este lamda
        if(porc > bestPorc):
            bestLamda = lamdas[i]
            bestPorc = porc
            bestThetas = thetas

    # Nos guardamos los pesos óptimos
    print("··· Pesos guardados guardados en " + WEIGHTS_NAMES[0] + " ··· ")
    savemat(WEIGHTS_NAMES[0], mdict={
        'Theta1': bestThetas[0], 
        'Theta2': bestThetas[1],
        'Theta3': bestThetas[2],
        'Theta4': bestThetas[3],
        'Theta5': bestThetas[4],
    })

    # Log
    print("-> El OneVsAll tiene una precisión sobre validación del " + str(bestPorc) + "% con lamda = " + str(bestLamda) + " <-")

def TestLogisticRegression(X_test, y_test):
    '''
    Prueba el clasificador logístico sobre el conjunto de tests,
    cogiendo los pesos ya entrenados.
    '''
    # Cargamos los thetas óptimos del archivo (son 5, uno por cada flor)
    weights = loadmat(WEIGHTS_NAMES[0])
    num_etiquetas = 5 #TODO: cambiarlo
    thetaOpt = np.zeros((num_etiquetas, X_test.shape[1] + 1))
    for i in range(num_etiquetas):
        thetaOpt[i] = weights['Theta' + str(i+1)]

    # Comprobamos la efectividad
    print("··· Comprobando la precisión sobre el conjunto de test ··· ")
    m = X_test.shape[0]
    unosX = np.hstack([np.ones([m, 1]), X_test])
    z = sigmoid(hMatrix(unosX, thetaOpt))
    porc = calcula_porcentaje(y_test, z, 4)

    # Log
    print("-> El OneVsAll tiene una precisión sobre test del " + str(porc) + "% <-")


# Red neuronal de 2 capas #
def NeutralNetworkClassifier(data:dict, ocultas:tuple, lamdas:tuple, iters:tuple):
    '''
    Clasificador por red neuronal de 2 capas 
    Recibe el término de regularización, el número de capas ocultas 
    y las iteraciones máximas para el descenso de gradiente
    '''
    # 1. Montamos la red neuronal
    # Atributos
    num_entradas = IMG_SIZE * IMG_SIZE * 3 # 3 por cada píxel (R, G, B)
    # num_ocultas = nos la pasan como parámetro
    num_etiquetas = len(FLOWER_NAMES) # 5

    # Sacamos los datos del diccionario
    X_train = data["X_train"]
    y_train_onehot = makeOneHot(data["y_train"].ravel(), num_etiquetas)
    X_val = data["X_val"]
    y_val = data["y_val"].ravel()
    m = X_train.shape[0]

    # Mejores parámetros para la red
    bestOcultas = ocultas[0]
    bestLamda = lamdas[0]
    bestIters = iters[0]
    bestThetas = []
    bestPorc = 0

    print("··· Entrenando la red neuronal (puede tardar varios lustros) ··· ")
    # Barra de progreso 
    pbar = tqdm(total= (len(ocultas) * len(lamdas) * len(iters)))

    # Probamos todas las combinaciones de valores que nos pasan (TARDA MUCHO)
    for i in range (len(ocultas)):
        for j in range (len(lamdas)):
            for k in range(len(iters)):
                print(" Probando con " + str(ocultas[i]) + " ocultas, lamda = " + str(lamdas[j]) + " y " + str(iters[k]) + " it.")
                # La entrenamos y cogemos los pesos óptimos (es el código de la Práctica 4)
                theta1, theta2 = trainNeutralNetwork(num_entradas, ocultas[i], num_etiquetas, X_train, y_train_onehot, lamdas[j], iters[k])

                # 2. Con los pesos obtenidos, hacemos la propagación hacia delante y vemos el porcentaje sobre validación
                a1, z2, a2, z3, h = forward_prop(X_val, theta1, theta2) 
                porcentaje = calcula_porcentaje(y_val, h, 3)

                print("* " + str(porcentaje) + "% *")

                # Comprobamos que sea mejor el porcentaje
                if(porcentaje > bestPorc):
                    bestPorc = porcentaje
                    bestOcultas = ocultas[i]
                    bestLamda = lamdas[j]
                    bestIters = iters[k]
                    bestThetas = [theta1, theta2]

                pbar.update(1) #Actualizar el GUI

    # Guardamos los pesos óptimos
    print("··· Pesos guardados guardados en " + WEIGHTS_NAMES[1] + " ··· ")
    savemat(WEIGHTS_NAMES[1], mdict={
        'Theta1': theta1, 
        'Theta2': theta2})

    # Sacamos el porcentaje de aciertos
    print("-> La red tiene una precisión sobre validacion del " +  str(bestPorc) +  " % con " + str(bestOcultas) + " capas ocultas, lamda = " + str(bestLamda) + " y " + str(bestIters) + " iteraciones <-")


def TestNeutralNetwork(X_test, y_test):
    '''
    Prueba la red neuronal sobre el conjunto de tests,
    cogiendo los pesos ya entrenados.
    '''

    # Cargamos los pesos óptimos desde la matriz
    print("··· Pesos guardados guardados en " + WEIGHTS_NAMES[1] + " ··· ")
    weights = loadmat(WEIGHTS_NAMES[1])
    theta1 = weights['Theta1']
    theta2 = weights['Theta2']

    # 2. Con los pesos óptimos obtenidos, hacemos la propagación hacia delante y obtenemos la predicción de la red
    print("··· Comprobando la precisión sobre el conjunto de test ··· ")
    a1, z2, a2, z3, h = forward_prop(X_test, theta1, theta2) 

    # Sacamos el porcentaje de aciertos
    porcentaje = calcula_porcentaje(y_test, h, 3)
    print("-> La red tiene una precisión del ",  porcentaje, " % <-")


# Support Vector Machines #
def SVMClassifier(data:dict, kernelType:str, values:list):

    # Sacamos los datos del diccionario
    X_train = data["X_train"]
    y_train = data["y_train"].ravel()
    X_val = data["X_val"]
    y_val = data["y_val"].ravel()

    #Cogemos el mejor modelo
    print("··· Entrenando la SVM (puede tardar varios minutos) ··· ")
    svm, cOpt, sigmaOpt = getBestSVMMultiClass(kernelType, values, X_train, y_train, X_val, y_val)

    #Guardamos la SVM
    print("··· SVM guardada en " + WEIGHTS_NAMES[2] + " ··· ")
    dump(svm, WEIGHTS_NAMES[2])
    
    # Vemos el porcentaje de aciertos sobre el conjunto de tests
    print("··· Comprobando la precisión sobre el conjunto de validación con C = " + str(cOpt) + ", sigma = " + str(sigmaOpt) + " ··· ")
    h = svm.predict(X_test)
    porcentaje = calcula_porcentaje_Y(y_test, h, 4)

    print("-> La SVM tiene una precisión del ",  porcentaje, "% <-")

def TestSVMClassifier(X_test, y_test):
    '''
    Prueba la red neuronal sobre el conjunto de tests,
    cogiendo los pesos ya entrenados.
    '''

    #Cargamos
    svmMultiClass = load(WEIGHTS_NAMES[2])

    # Vemos el porcentaje de aciertos sobre el conjunto de tests
    print("··· Comprobando la precisión sobre el conjunto de test ··· ")
    h = svmMultiClass.predict(X_test)
    porcentaje = calcula_porcentaje_Y(y_test, h, 4)

    print("-> La SVM tiene una precisión del ",  porcentaje, "% <-")

def SaveSets(datasets:tuple):
    '''
    Guarda los datasets ya separados en un archivo de MatLab 
    para no tener que cargar las imágenes todo el rato
    '''
    savemat(DATA_FILENAME, mdict={
        'X_train': datasets[0], 
        'y_train': datasets[1], 
        'X_val': datasets[2], 
        'y_val': datasets[3], 
        'X_test': datasets[4],
        'y_test': datasets[5]})

    print("··· Datasets guardados en " + DATA_FILENAME +  " ··· ")


# 1a). Cargamos todas las imágenes de sus respectivas carpetas y guardamos los datasets 
# (solo hay que llamar a esto una vez)
'''
datasets = LoadAllImages()
SaveSets(datasets)
'''
# Parámetros para ajustar los modelos con el set de validación
lamdas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]#, 100, 300] # Para los lambdas de NN y los {C, sigma} de SVM
ocultas = [25, 50, 100] 
iters = [70, 150, 250]


# 1b). Cargamos los datasets ya preparados

data = loadmat(DATA_FILENAME)
X_test = data['X_test']
y_test = data['y_test'].ravel()
X_train = data['X_train']
y_train = data['y_train'].ravel()

# 2. Llamamos al clasificador que sea
LogisticRegressionClassifier(data, [100]) #lamda = 100 -> 44.5% val. (93% train)
TestLogisticRegression(X_test, y_test)

# NeutralNetworkClassifier(data, [25], [1], [210]) #ocultas = 25, lamda = 1, iters = 140 -> 43% test. (93% train)
#TestNeutralNetwork(X_test, y_test)

#SVMClassifier(data, 'rbf', lamdas) # C = 3, sigma = 30 -> 53% val. (98% train)
#TestSVMClassifier(X_test, y_test)

#Cosas que afectan al porcentaje de aciertos:
# 1. Hiperparámetros (lamda, num_ocultas, num_iter, C, sigma)
# 2. Tamaño del conjunto de entrenamiento 
# 3. Tamaño de las imágenes 


#TODO: 
# 1. Encontrar los parámetros óptimos (y hacer gráficas)
# 2. Usar optimize en vez de fmin_tnc en el OneVsAll para poder elegir el número de iteraciones