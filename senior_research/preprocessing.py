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