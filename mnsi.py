# -- coding: utf-8 --

# pip install pydot
# pip install graphviz
# Graphviz - Graph Visualization Software
# https://graphviz.gitlab.io/download/
"""
@author: IVAN
"""

# https://www.codificandobits.com/blog/tutorial-clasificacion-imagenes-redes-convolucionales-python/
#to import data + preprocessing
# import tensorflow as tf
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
# %matplotlib inline#to build the CNN
# import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D #test on a new image
import imageio
from tensorflow.keras.optimizers import Adam, SGD
from matplotlib import pyplot
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
import time

from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
import seaborn as sns
# import numpy as np
# from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#show one image with label
print(y_train[0])
plt.imshow(X_train[0], cmap='Greys')
plt.show()



#normalize image data only
#DEPRECATED: normalize, identical to the new method
# import numpy as np
X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
X_train /= 255.0
X_test /= 255.0

# one-hot encoding
nclases = 10
y_train = to_categorical(y_train, nclases)
y_test = to_categorical(y_test, nclases)

#change shape to IMAGES only for CNN input
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1) # over 14%
#the labels are already in an acceptable shape

# model topology (CNN)
# feature extraction
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
# headfrom tensorflow.keras.layers import Dropout

model.add(Dense(120, activation='relu'))   # Capa densa con 120 neuronas
model.add(Dropout(0.5))   
model.add(Dense(84,activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(nclases,activation='softmax'))

#   summarize model_1
#input layer not included
model.summary()

#   summarize model_2
#input layer is included
# from keras.utils.vis_utils import plot_model
# plot_model(model, show_shapes=True, show_layer_names=True)

opt = Adam(learning_rate = 1e-3) # by default lr=1e-3 
# opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='categorical_crossentropy',
      optimizer=opt,
      metrics=['accuracy'])
batch_size = 128
epochs = 10

inicio = time.time()

# training of the model
history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

fin = time.time()
print('\nTraining time (seg): %.2f' % (fin-inicio)) 
print('\n')

test_score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', test_score[0])
print('Test accuracy: %.2f%%' % (test_score[1]*100))

# Matriz de confusion
# Predecir todas las etiquetas de los datos de prueba
test_predictions = model.predict(X_test)
test_predictions = np.argmax(test_predictions, axis=1)
test_true_labels = np.argmax(y_test, axis=1)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(test_true_labels, test_predictions)

# Imprimir la matriz de confusión
print(conf_matrix)

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# plot loss during training
pyplot.title('Loss / categorical_crossentropy')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
# plot accuracy during training
pyplot.title('Accuracy / categorical_crossentropy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()


#predict a digit in x_test
prediction = model.predict(X_test[0].reshape(1, 28, 28, 1))
print("\nactual digit: ", y_test[0], " predicted: ", prediction.argmax())


# Test it on an image available on PC (current folder)
im = imageio.imread("a3Rql9C.png")
#visualize image
gray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()

# reshape the image
gray = gray.reshape(1, 28, 28, 1)
# normalize image
gray /= 255

prediction = model.predict(gray.reshape(1, 28, 28, 1))
print("predicted number (png): ", prediction.argmax())

# Al final de tu código de entrenamiento
model.save('models/model_Mnist_LeNet.h5')
print("Saved model to disk")