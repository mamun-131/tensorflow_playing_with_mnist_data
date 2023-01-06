# MNIST DATASET, TENSOWFLOW, KERAS, SEQUENTIAL MODEL, SIMPLE NEURAL NETWORK, MULTIPLE DENSE LAYERS

<details><summary>Table of Contents</summary><p>

* [Introduction](#introduction)
* [Get, split and shape the Data](#get-split-and-shape-the-data)
* [Model Types](#usage)
* [Sequential Model](#sequential-model)
* [Functional Model](#functional-model)
</p></details><p></p>


## Introduction

I have played with MNIST dataset using Tensowflow and Keras. I have coded for sequential and functional, Neural Networks in different files. This Neural Network 
is simple. It has just Flatten Layer, 2 Dense Layers and Output Layers. 

At the starting part of the coding we need to call the libraries.

Code:
```
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
```
## Get, split and shape the Data
Following code doenload MNIST dataset and split training and testing data

Code:
```
mydata = tf.keras.datasets.mnist

(x_train, y_train), (x_test,y_test) = mydata.load_data()
print (x_train.shape)
x_train, x_test = x_train/255, x_test/255
```
## Model Types
You can cretae model by two ways.
- **Sequential Way** - Sequential Model.
- **Functional Way** - Functional Model.

## Sequential Model
Sequential Model Code should be like below. I have created a separate file for Sequential Model (tf_mnist_sequential_model.py).

Code:
```
--- Create a sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')    
    ]) 
```
## Functional Model
Functional Model Code should be like below. I have created a separate file for Functional Model (tf_mnist_functional_model.py).

Code:
```
--- Create a Functional model
--- Functional model has 3 parts. Input, Layers and Model

- Define input to the model
input_layer = tf.keras.layers.Input(shape=(28,28))

- Define a set of interconnected layers on the input
flatten_layer = tf.keras.layers.Flatten()(input_layer)
dense_layer1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(flatten_layer)
dense_layer2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(dense_layer1)
output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(dense_layer2)

- Define the Model using input and output layers
functional_model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

functional_model.summary()
```
