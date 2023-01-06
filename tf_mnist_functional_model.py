# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


mydata = tf.keras.datasets.mnist

(x_train, y_train), (x_test,y_test) = mydata.load_data()
print (x_train.shape)
#print(x_train[0])

# for i in range(5):
#     plt.imshow(x_train[i])
#     plt.show()

# plt.imshow(x_train[0])
# plt.show()

x_train, x_test = x_train/255, x_test/255

### Create a Functional model
### Functional model has 3 parts. Input, Layers and Model

# Define input to the model
input_layer = tf.keras.layers.Input(shape=(28,28))

# Define a set of interconnected layers on the input
flatten_layer = tf.keras.layers.Flatten()(input_layer)
dense_layer1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(flatten_layer)
dense_layer2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(dense_layer1)
output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(dense_layer2)

# Define the Model using input and output layers
functional_model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

functional_model.summary()

### Compile the model
functional_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

print("-----------------train and save the model--------------------")
### Train the model
functional_model.fit(x_train, y_train, epochs=5)
functional_model.save('mnist_functional_model.h5')
### Test the model
print("-----------------test the model--------------------")
functional_model.evaluate(x_test, y_test)

print("-----------------load trained model--------------------")
trained_model = tf.keras.models.load_model('mnist_functional_model.h5')
acc = trained_model.evaluate(x_test, y_test, verbose=0)
print('evalution---' + str(acc))

print("-----------------load new testing image--------------------")
my_image = tf.keras.preprocessing.image.load_img('sample_img1.png', color_mode = 'grayscale', target_size=(28,28))
my_image = tf.keras.preprocessing.image.img_to_array(my_image)
my_image = my_image.reshape(1, 28, 28, 1)
my_image = my_image.astype('float32')
my_image = my_image/255.0

print("-----------------prediction with new testing image--------------------")
prediction = trained_model.predict(my_image)
digit = np.argmax(prediction)
print("the digit in the file/image is " + str(digit))