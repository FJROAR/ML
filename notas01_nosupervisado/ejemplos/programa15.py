import numpy as np
import matplotlib.pyplot as plt



#En este caso para cargar los datos se hace uso de una función específica creada
#para tal fin: load_data()

def load_data():
    X = np.load("data/X_part1.npy")
    X_val = np.load("data/X_val_part1.npy")
    y_val = np.load("data/y_val_part1.npy")
    return X, X_val, y_val


X_train, X_val, y_val = load_data()

# Display the first five elements of X_train
print("The first 5 elements of X_train are:\n", X_train[:5])  

# Display the first five elements of X_val
print("The first 5 elements of X_val are\n", X_val[:5])  

# Display the first five elements of y_val
print("The first 5 elements of y_val are\n", y_val[:5]) 


print ('The shape of X_train is:', X_train.shape)
print ('The shape of X_val is:', X_val.shape)
print ('The shape of y_val is: ', y_val.shape)



#Visualización de la información

plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b') 

# Set the title
plt.title("The first dataset")
# Set the y-axis label
plt.ylabel('Throughput (mb/s)')
# Set the x-axis label
plt.xlabel('Latency (ms)')
# Set axis range
plt.axis([0, 30, 0, 30])
plt.show()



#Estimación de los  parámetros de la gaussiana multivariante con supuesto de
#independiencia entre las variables y por tanto las covarianzas son 0

def estimate_gaussian(X): 
    """
    Calculates mean and variance of all features 
    in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """

    m, n = X.shape
    
    mu = np.mean(X, axis = 0)
    var = np.var(X, axis = 0)
        
    return mu, var


mu_train, var_train = estimate_gaussian(X_train)              

print("Mean of each feature:", mu_train)
print("Variance of each feature:", var_train)


def multivariate_gaussian(X, mu, var):
    """
    Computes the probability 
    density function of the examples X under the multivariate gaussian 
    distribution with parameters mu and var. If var is a matrix, it is
    treated as the covariance matrix. If var is a vector, it is treated
    as the var values of the variances in each dimension (a diagonal
    covariance matrix
    """
    
    k = len(mu)
    
    if var.ndim == 1:
        var = np.diag(var)
        
    X = X - mu
    p = (2* np.pi)**(-k/2) * np.linalg.det(var)**(-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, 
                             axis=1))
    
    return p
   


p = multivariate_gaussian(X_train, mu_train, var_train)


#Visualización de las curvas de nivel asociada a los datos. Para un nuevo
#set de datos habría que estudiar las variables para establecer bien esta
#visualización que para un caso sencillo de 2 variables, resulta interesante

     
def visualize_fit(X, mu, var):
    """
    This visualization shows you the 
    probability density function of the Gaussian distribution. Each example
    has a location (x1, x2) that depends on its feature values.
    """
    
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariate_gaussian(np.stack([X1.ravel(), 
                                        X2.ravel()], 
                                       axis=1), 
                              mu, 
                              var)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx')

    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, 
                    X2, 
                    Z, 
                    levels=10**(np.arange(-20., 1, 3)), 
                    linewidths=1)
        
    # Set the title
    plt.title("The Gaussian contours of the distribution fit to the dataset")
    # Set the y-axis label
    plt.ylabel('Throughput (mb/s)')
    # Set the x-axis label
    plt.xlabel('Latency (ms)')
    

visualize_fit(X_train, mu_train, var_train)



#Selección del punto de corte: 
    #En este caso se hace uso del conjunto X_val que son observaciones reales
    #Donde se conoce y_val
    #Y por tanto se puede maximizar el F1-score

def select_threshold(y_val, p_val): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), 
                             max(p_val), 
                             step_size):
    
        predictions = (p_val < epsilon)
        
        tp = sum((predictions == 1) & (y_val == 1))
        fp = sum((predictions == 1) & (y_val == 0))
        fn = sum((predictions == 0) & (y_val == 1))
        
        prec = tp  / (tp + fp)
        rec = tp / (tp + fn)
        
        F1 = 2 * prec * rec / (prec + rec)
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1


p_val = multivariate_gaussian(X_val, mu_train, var_train)
epsilon, F1 = select_threshold(y_val, p_val)
print("El valor de epsilon es: ", epsilon)

#Se observa el efecto en X_train
# Find the outliers in the training set 
outliers = p < epsilon

#Con los parámetros de X_train

# Visualize the fit
visualize_fit(X_train, mu_train, var_train)

# Draw a red circle around those outliers
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
         markersize= 10,markerfacecolor='none', markeredgewidth=2)
