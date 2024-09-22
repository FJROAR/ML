#IMDB dataset: a set of 50,000 highly polarized reviews from the
#Internet Movie Database

import pandas as pd

from keras.datasets import imdb
(train_data, train_labels), (test_data, 
                             test_labels) = imdb.load_data(num_words=10000)


#El conjunto de datos tienen como entrada índices de las palabras más frecuentes
#la combinación de estos índices definirán la crítica positiva o negativa
#de la película 1 o 0

train_data[0]


#Decodificación de un review

word_index = imdb.get_word_index()
reverse_word_index = dict(
[(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
[reverse_word_index.get(i - 3, '?') for i in train_data[1]])

decoded_review


#Preparación de datos

import numpy as np

#Se construyen vectores de 10000 posiciones por elemento de entrenamiento y 
#consecuentemente de test One-hot encode your lists to turn them into 
#vectors of 0s and 1s
def vectorize_sequences(sequences, dimension=10000):
    
    #sequences = train_data
    
    results = np.zeros((len(sequences), dimension))
    
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


#Se construye el modelo de red

from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


#Compilación


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])



#Introducción de un conjunto de validación durante el entrenamiento

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=4,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys()

acc = history_dict['acc']
val_acc = history_dict['val_acc']

import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#training vs acc
#plt.clf()
#acc_values = history_dict['acc']
#val_acc_values = history_dict['val_acc']
#plt.plot(epochs, acc, 'bo', label='Training acc')
#plt.plot(epochs, val_acc, 'b', label='Validation acc')
#plt.title('Training and validation accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()


#In this case, to prevent overfitting, you could stop training after three epochs. 
#This fairly naive approach achieves an accuracy of 88%. With state-of-the-art
#approaches, you should be able to get close to 95%

#Predicciones en el conjunto test o en algún elemento que se introduzca
model.predict(x_test)

#As you can see, the network is confident for some samples (0.99 or more, or 0.01 or
#less) but less confident for others (0.6, 0.4)

df_final = pd.DataFrame(x_test)
df_final['real'] = y_test
df_final['pred'] = model.predict(x_test)

df_final.head(20)

model.summary()