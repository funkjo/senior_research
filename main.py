from senior_research import preprocessing as pre
# from senior_research import model as m
import tensorflow as tf
import numpy as np


def main():
    images, labels = pre.convert_img2array()
    train_images, test_images, train_labels, test_labels = pre.create_train_test_sets(images, labels, 0.2)

    print("images shape ", images.shape)
    print("labels shape ", labels.shape)
    print("train_images shape ", train_images.shape)
    print("test_images shape ", test_images.shape)
    print("singe image shape ", train_images[0].shape)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(120,120)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)
    

if __name__ == "__main__":
    main()