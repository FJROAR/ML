import numpy as np
import matplotlib.pyplot as plt

# Se lee la imagen
original_img = plt.imread('data/20221109_marte_ex.jpg')


# Visualización
plt.imshow(original_img)


#Se observa el tamaño del fichero
print("Shape of original_img is:", original_img.shape)


#Se normalizan los datos
original_img = original_img / 255

#Se ajusta la imagen para que se le pueda aplicar el kmeans
# Reshape the image into an m x 3 matrix where m = number of pixels
# (in this case m = 176 x 172 = 30272)
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X_img that we will use K-Means on.

X_img = np.reshape(original_img, 
                   (original_img.shape[0] * original_img.shape[1], 
                    3)
                   )

print("Shape of X_img is:", X_img.shape)

#Lo que se quiere es agrupar filas parecidas según sus píxeles, claramente, 
#esto sería una simplificación en la riqueza de los píxeles que se considera
#en la foto original y por tanto se tendrá tanto valores diferentes como K se
#haya elegido ya que a cada punto se le asignará el centroide final del Kmeans

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
#K = 8
#max_iters = 10               
max_iters = 20               

#Se inicializan los centroides tomando K puntos de modo aleatorio

def kMeans_init_centroids(X, K, seed = 155):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    Args:
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    
    Returns:
        centroids (ndarray): Initialized centroids
    """
    
    # Randomly reorder the indices of examples
    np.random.seed(seed)
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids

 
initial_centroids = kMeans_init_centroids(X_img, K) 



#Se ejecuta el algoritmo K-Means e interesa asociar cada dato a su centroide

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): k centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """


    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        
        distance = []
        
        for j in range(centroids.shape[0]):
            norm_ij =  np.linalg.norm(X[i] - centroids[j])
            distance.append(norm_ij)
            
        idx[i] = np.argmin(distance) 
    
    return idx

def compute_centroids(X, idx, K):
    
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    
    for k in range(K):
        points = X[idx == k]
        centroids[k] = np.mean(points, axis = 0)
    
    return centroids


def run_kMeans(X, initial_centroids, max_iters=10):
    
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(m)
    
    # Run K-Means
    for i in range(max_iters):
        
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
        
    plt.show() 
    
    return centroids, idx


centroids, idx = run_kMeans(X_img, initial_centroids, max_iters) 


#Se recupera la imagen

# Represent image in terms of indices
X_recovered = centroids[idx, :] 

# Reshape recovered image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img.shape) 


#Se representa el resultado final y se compara con el original

fig, ax = plt.subplots(1,2, figsize=(8,8))
plt.axis('off')

ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].set_axis_off()


# Display compressed image
ax[1].imshow(X_recovered)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()