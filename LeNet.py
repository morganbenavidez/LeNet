import tensorflow as tf
import keras
from keras import models, layers

# input shape = (227, 227, 3)
def LeNet(input_shape, num_classes):

    model = models.Sequential()

    # Layer 1
    model.add(layers.Conv2D(6, (5, 5), strides=(1, 1), input_shape=input_shape, padding='same', activation='tanh'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Layer 2
    model.add(layers.Conv2D(16, (5, 5), strides=(1, 1), padding='valid', activation='tanh'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Flatten the output for the fully connected layers
    model.add(layers.Flatten())

    # Fully connected layers
    # First Dense Layer
    model.add(layers.Dense(120, activation='tanh'))

    # Second Dense Layer
    model.add(layers.Dense(84, activation='tanh'))

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

Model = LeNet((32, 32, 1), 10)
print(Model.summary())