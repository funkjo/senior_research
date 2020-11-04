# Data Preprocessing
# John Funk
# 4 November 2020


# import

import PIL
from PIL import Image 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array 
import numpy as np 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def convert_img2array():
    """
    This function goes through each directory in the data folder and converts images to numpy arrays
    Returns 2 numpy arrays: images converted into numpy arrays, corresponding labels
    """

    images = []
    labels = []
    directories = ["senior_research/data/Car", "senior_research/data/Crosswalk", "senior_research/data/Hydrant", "senior_research/data/TrafficLight"]

    for folder in directories:
        if "Car" or "Crosswalk" in folder.path:
            i = 0
            for entry in os.scandir(folder):
                if i == 1000:
                    i = 0
                    break
                if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
                    img = load_img(entry.path)
                    converted = img_to_array(img)
                    images.append(converted)
                    labels.append(0)
                    i += 1
        else:
            for entry in folder:
                if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
                    img = load_img(entry.path)
                    converted = img_to_array(img)
                    images.append(converted)
                    labels.append(0)

    images = np.asarray(images)
    labels = np.asarray(labels)

    return images, labels