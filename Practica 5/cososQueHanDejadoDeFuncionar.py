
def h(x, theta):
    return np.dot(x, theta[np.newaxis].T)

def coste(X, Y, Theta):
    #Calculamos la función de coste
    m = X.shape[0]
    sum1 = 0
    for k in range(0, m):
        sum1 += (h(X[k], Theta) - Y[k])**2
    return 1 / (2*m) * sum1

def costeRegularizado(X, Y, theta, lamda):
    c = coste(X, Y, theta)
    n = theta.shape[0]
    m = X.shape[0]

    sum2 = 0
    for k in range(1, n):
        sum2 += theta[k]**2
    
    aux = (lamda / (2*m))*sum2
    return c + aux

def gradienteRegularizado(X, Y, theta, lamda):
    t = theta.shape
    gradReg = np.zeros(t)
    m = X.shape[0]
    
    sum0 = 0
    for i in range(0, m):
        sum0 += (h(X[i], theta) - Y[i])*(X[i, 0])
    gradReg[0] = (1/m)*(sum0)

    sum1 = 0
    # n =
    # for j in range(1, n)
    for i in range(0, m):
        sum1 += (h(X[i], theta) - Y[i])*(X[i, 1])
    gradReg[1] = ((1/m)*(sum1)) + ((lamda/m)*theta[1])
    
    return gradReg