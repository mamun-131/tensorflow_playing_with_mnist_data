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

### Create a sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')    
    ]) 

model.summary()

### Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

print("-----------------train and save the model--------------------")
### Train the model
model.fit(x_train, y_train, epochs=5)
model.save('mnist_sequential_model.h5')
### Test the model
print("-----------------test the model--------------------")
model.evaluate(x_test, y_test)

print("-----------------load trained model--------------------")
trained_model = tf.keras.models.load_model('mnist_sequential_model.h5')
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