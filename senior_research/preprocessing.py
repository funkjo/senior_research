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
import cv2


def convert_img2array():
    """
    This function goes through each directory in the data folder and converts images to numpy arrays
    Returns 2 numpy arrays: images converted into numpy arrays, corresponding labels
    """

    images = []
    labels = []
    classes = ['Car', 'Crosswalk', 'Hydrant', 'TrafficLight']
    datadir = '/Users/johnfunk/OneDrive - Eastern Connecticut State University/Courses/Computer Science/Senior-Research/github-project/senior_research/senior_research/data'
    directories = ["senior_research/data/Car", "senior_research/data/Crosswalk", "senior_research/data/Hydrant", "senior_research/data/TrafficLight"]

    # for folder in directories:
    #     if "Car" or "Crosswalk" in folder.path:
    for _class in classes:
        path = os.path.join(datadir, _class)
        if _class == 'Car' or _class == 'Crosswalk':
            i = 0
            # for entry in os.scandir(folder):
            for img in os.listdir(path):
                if i == 1000:
                    i = 0
                    break
                # if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
                # img = load_img(entry.path)
                # converted = img_to_array(img)
                converted = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                images.append(converted)
                if _class == 'Car':
                    labels.append(0)
                else:
                    labels.append(1)
                i += 1
        else:
            # for entry in folder:
            for img in os.listdir(path):
                # if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
                # img = load_img(entry.path)
                # converted = img_to_array(img)
                converted = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                images.append(converted)
                if _class == 'Hydrant':
                    labels.append(2)
                else:
                    labels.append(3)

    images = np.asarray(images).reshape(len(images), 120, 120)
    labels = np.asarray(labels)

    images2 = images / 255.0

    return images2, labels


def create_train_test_sets(images, labels, _test_size):
    """
    Uses scikit-learns train_test_split function to split the data into a training set and test set
    Returns training images, test images, training labels, test labels
    """
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=_test_size, random_state=32)

    return X_train, X_test, y_train, y_test