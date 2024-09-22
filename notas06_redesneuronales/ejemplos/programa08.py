from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images.shape
len(train_labels)
train_labels

test_images.shape
len(test_labels)

test_labels

#Arquitectura de red

from keras import models
from keras import layers



network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

#Se preparan los datos de modo que tengan las dimensiones de entrada adecuadas

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

#Análogamente se hace con las etiquetas

from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


#Se entrena la red

history = network.fit(train_images, train_labels, epochs=5, batch_size=128)

history_dict = history.history
history_dict.keys()

loss = history_dict['loss']
accuracy = history_dict['accuracy']

import matplotlib.pyplot as plt

epochs = range(1, len(loss) + 1)
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Evaluación del modelo en el conjunto de prueba
test_loss, test_accuracy = network.evaluate(test_images, test_labels)

print(f'Test accuracy: {test_accuracy}')


#Y se pueden ver los parámetros con

network.summary()