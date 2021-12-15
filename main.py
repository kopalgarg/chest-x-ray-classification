import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from keras.layers import LeakyReLU
from tensorflow.keras.optimizers import SGD
import numpy.linalg as LA
from sklearn import datasets, svm, metrics
import cv2
from PIL import Image
from sklearn.utils import shuffle
import tensorflow as tf
tf.test.gpu_device_name()
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

# setting data paths
ROOT = "."
data = "./../data/out"

# load images
train = loaddata('./../data/out/train')
valid = loaddata('./../data/out/valid')
test = loaddata('./../data/out/test')

# shuffle
train = shuffle(train)
valid = shuffle(valid)
test = shuffle(valid)

print(train.shape)
print(valid.shape)
print(test.shape)

# preprocessing

X_train,Y_train = flatImages(train, selectedLabels=[0,1,2])
X_test,Y_test = flatImages(test, selectedLabels=[0,1,2])
X_valid,Y_valid = flatImages(valid, selectedLabels=[0,1,2])

X_train = X_train.reshape(-1,X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(-1,X_test.shape[1]*X_test.shape[2])
X_valid = X_valid.reshape(-1,X_valid.shape[1]*X_valid.shape[2])

# PCA
pca = PCA(n_components=1024, random_state=2515)
projected_X_train = pca.fit_transform(X_train)
projected_X_valid = pca.transform(X_valid)
projected_X_test = pca.transform(X_test)

