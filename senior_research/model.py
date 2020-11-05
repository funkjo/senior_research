# Model Building
# John Funk
# 4 November 2020


# import
import tensorflow as tf
import numpy as np


def create_model():
    """
    Creates a model and sets up the layers
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(3,120,120)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model






