# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


mydata = tf.keras.datasets.mnist

(x_train, y_train), (x_test,y_test) = mydata.load_data()
print (x_train.shape)
#print(x_train[0])

# for i in range(5):
#     plt.imshow(x_train[i])
#     plt.show()

plt.imshow(x_train[0])
plt.show()