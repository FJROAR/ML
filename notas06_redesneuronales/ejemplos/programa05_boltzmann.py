#https://github.com/fawazsiddiqi/recommendation-system-with-a-Restricted-Boltzmann-Machine-using-tensorflow/blob/master/notebooks/CollaborativeFilteringUsingRBM.ipynb

#Tensorflow library. Used to implement machine learning models
import tensorflow as tf
#Numpy contains helpful functions for efficient mathematical calculations
import numpy as np
#Dataframe manipulation library
import pandas as pd
#Graph plotting library
import matplotlib.pyplot as plt


#Loading in the movies dataset
movies_df = pd.read_csv('data/ml-1m/movies.dat', 
                        sep='::', 
                        header=None, 
                        engine='python',
                        encoding= 'Latin1')
movies_df.head()


#Loading in the ratings dataset
ratings_df = pd.read_csv('data/ml-1m/ratings.dat', 
                         sep='::', 
                         header=None, 
                         engine='python',
                         encoding= 'Latin1')
ratings_df.head()

movies_df.columns = ['MovieID', 'Title', 'Genres']
movies_df.head()

ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings_df.head()


#Restricte Boltzmann Machine

len(movies_df)

#Normalización de la información
user_rating_df = ratings_df.pivot(index='UserID', 
                                  columns='MovieID', 
                                  values='Rating')
user_rating_df.head()

norm_user_rating_df = user_rating_df.fillna(0) / 5.0
trX = norm_user_rating_df.values
trX[0:5]

#Parámetros del modelo

hiddenUnits = 20
visibleUnits =  len(user_rating_df.columns)

vb = tf.Variable(tf.zeros([visibleUnits]), 
                 tf.float32) #Number of unique movies

hb = tf.Variable(tf.zeros([hiddenUnits]), 
                 tf.float32) #Number of features we're going to learn

W = tf.Variable(tf.zeros([visibleUnits, 
                          hiddenUnits]), 
                tf.float32)

v0 = tf.zeros([visibleUnits], tf.float32)

#testing to see if the matrix product works
tf.matmul([v0], W)

#Phase 1: Input Processing
#defining a function to return only the generated hidden states 
def hidden_layer(v0_state, W, hb, seed):
    
    h0_prob = tf.nn.sigmoid(tf.matmul([v0_state], W) + 
                            hb)  #probabilities of the hidden units
    tf.random.set_seed(seed)
    h0_state = tf.nn.relu(tf.sign(h0_prob - 
                                  tf.random.uniform(tf.shape(h0_prob)))) #sample_h_given_X
    return h0_state

#printing output of zeros input

h0 = hidden_layer(v0, W, hb, 155)
print("first 15 hidden states: ", 
      h0[0][0:15])

def reconstructed_output(h0_state, W, vb, seed):
    
    v1_prob = tf.nn.sigmoid(tf.matmul(h0_state, 
                                      tf.transpose(W)) + vb) 
    tf.random.set_seed(seed)
    v1_state = tf.nn.relu(tf.sign(v1_prob - 
                                  tf.random.uniform(tf.shape(v1_prob)))) #sample_v_given_h
    return v1_state[0]


v1 = reconstructed_output(h0, W, vb, 155)
print("hidden state shape: ", h0.shape)
print("v0 state shape:  ", v0.shape)
print("v1 state shape:  ", v1.shape)

#Función de error
def error(v0_state, v1_state):
    
    return tf.reduce_mean(tf.square(v0_state - v1_state))

err = tf.reduce_mean(tf.square(v0 - v1))

print("error" , err.numpy())

#Entrenamiento

epochs = 5
batchsize = 500
errors = []
weights = []
K=1
alpha = 0.1

#creating datasets
train_ds = \
    tf.data.Dataset.from_tensor_slices((np.float32(trX))).batch(batchsize)



#for i in range(epochs):
#    for start, end in zip( range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
#        batch = trX[start:end]
#        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
#        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
#        cur_nb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
#        prv_w = cur_w
#        prv_vb = cur_vb
#        prv_hb = cur_hb
#    errors.append(sess.run(err_sum, feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
#    print (errors[-1])
v0_state=v0

for epoch in range(epochs):
    batch_number = 0
    for batch_x in train_ds:

        for i_sample in range(len(batch_x)):           

            for k in range(K):
                
                v0_state = batch_x[i_sample]
                
                h0_state = hidden_layer(v0_state, 
                                        W, 
                                        hb, 
                                        i_sample + epochs + k)
                
                v1_state = reconstructed_output(h0_state, 
                                                W, 
                                                vb, 
                                                i_sample + epochs + k)
                
                h1_state = hidden_layer(v1_state, 
                                        W, 
                                        hb, 
                                        i_sample + epochs + k)

                delta_W = tf.matmul(tf.transpose([v0_state]), 
                                    h0_state) - tf.matmul(tf.transpose([v1_state]), 
                                                          h1_state)
                W = W + alpha * delta_W

                vb = vb + alpha * tf.reduce_mean(v0_state - v1_state, 0)
                hb = hb + alpha * tf.reduce_mean(h0_state - h1_state, 0) 

                v0_state = v1_state

            if i_sample == len(batch_x)-1:
                
                err = error(batch_x[i_sample], v1_state)
                errors.append(err)
                weights.append(W)
                print ( 'Epoch: %d' % (epoch + 1), 
                       "batch #: %i " % batch_number, "of %i" % (len(trX)/batchsize), 
                       "sample #: %i" % i_sample,
                       'reconstruction error: %f' % err)
        
        batch_number += 1


plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch x batch')
plt.show()


#Predicción

mock_user_id = 215

#Selecting the input user
inputUser = trX[mock_user_id-1].reshape(1, -1)

inputUser = tf.convert_to_tensor(trX[mock_user_id-1],"float32")
v0 = inputUser

print(v0)

v0.shape

v0test = tf.zeros([visibleUnits], tf.float32)
v0test.shape


#Feeding in the user and reconstructing the input

hh0 = tf.nn.sigmoid(tf.matmul([v0], W) + hb)

vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)

rec = vv1

tf.maximum(rec,1)
for i in vv1:
    print(i)
    

scored_movies_df_mock = movies_df[movies_df['MovieID'].isin(user_rating_df.columns)]
scored_movies_df_mock = scored_movies_df_mock.assign(RecommendationScore = rec[0])
scored_movies_df_mock.sort_values(["RecommendationScore"], ascending=False).head(20)

movies_df_mock = ratings_df[ratings_df['UserID'] == mock_user_id]
movies_df_mock.head()

#Merging movies_df with ratings_df by MovieID
merged_df_mock = scored_movies_df_mock.merge(movies_df_mock, 
                                             on='MovieID', 
                                             how='outer')

merged_df_mock.sort_values(["RecommendationScore"], ascending=False).head(20)