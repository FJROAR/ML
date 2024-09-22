import numpy as np
import matplotlib.pyplot as plt



#En este caso para cargar los datos se hace uso de una función específica creada
#para tal fin: load_data_multi()

def load_data_multi():
    X = np.load("data/X_part2.npy")
    X_val = np.load("data/X_val_part2.npy")
    y_val = np.load("data/y_val_part2.npy")
    return X, X_val, y_val


# load the dataset
X_train_high, X_val_high, y_val_high = load_data_multi()


print ('The shape of X_train_high is:', X_train_high.shape)
print ('The shape of X_val_high is:', X_val_high.shape)
print ('The shape of y_val_high is: ', y_val_high.shape)


#Se aplica el método de sklearn

from sklearn.mixture import GaussianMixture
import pandas as pd

#En este ejemplo se han modificado manualmente los parámetros n_components 
#(fundamentalmente), dejando fijo n_init = 5, para obtener un mejor ajuste en 
#términos de matriz de confusión respecto a la anterior simulación

modelo = GaussianMixture(n_components = 11, 
                         n_init = 5, 
                         random_state = 0).fit(X_train_high)

p_high2 = modelo.predict(X_train_high)

# Get the score for each sample
score = modelo.score_samples(X_train_high)
# Save score as a column
df = pd.DataFrame(X_train_high)
df['score'] = score
# Get the score threshold for anomaly
pct_threshold = np.percentile(score, 3)
# Print the score threshold

print(f'The threshold of the score is {pct_threshold:.2f}')
# Label the anomalies
df['anomaly_gmm_pct'] = df['score'].apply(lambda x: 
                                          1 if x < pct_threshold else 
                                          0)

anomaly = df['anomaly_gmm_pct']

score_val = modelo.score_samples(X_val_high)
df_val = pd.DataFrame(X_val_high)
df_val['score_val'] = score_val

#Se aplica el punto de corte del paso anterior para comparar

df_val['pred_val'] = df_val['score_val'].apply(lambda x: 
                                               1 if x < pct_threshold else 
                                               0)

    
from sklearn.metrics import confusion_matrix
confusion_matrix(y_val_high, df_val['pred_val'])



#Se repite lo anterior para el dataset de dimensión 2

def load_data():
    X = np.load("data/X_part1.npy")
    X_val = np.load("data/X_val_part1.npy")
    y_val = np.load("data/y_val_part1.npy")
    return X, X_val, y_val

X_train, X_val, y_val = load_data()


modelo0 = GaussianMixture(n_components = 2, 
                          n_init = 5, 
                          random_state = 0).fit(X_train)

# Get the score for each sample
score0 = modelo0.score_samples(X_train)
# Save score as a column
df0 = pd.DataFrame(X_train)
df0['score0'] = score0
# Get the score threshold for anomaly
pct_threshold0 = np.percentile(score0, 2)
# Print the score threshold
print(f'The threshold of the score is {pct_threshold0:.2f}')
# Label the anomalies
df0['anomaly_gmm_pct0'] = df0['score0'].apply(lambda x: 
                                              1 if x < pct_threshold0 else 
                                              0)


# Find the outliers in the training set 
outliers = score0 < pct_threshold0

# Visualize the fit

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
    

mu, var = estimate_gaussian(X_train)         


visualize_fit(X_train, mu, var)

# Draw a red circle around those outliers
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
         markersize= 10,markerfacecolor='none', markeredgewidth=2)