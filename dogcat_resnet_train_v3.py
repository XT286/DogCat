# Data Processing Libraries
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import h5py

# import from Keras Library
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Lambda
from keras.models import *
from keras.applications import *
from keras.applications.xception import preprocess_input


# Set constants
n_train = 25000
n_test = 12500
width = 224
test_dir = 'test/'
train_dir = 'train/'

# Load image, resize to 224*224, normalize the three channels with the mean values
X = np.zeros((n_train, width, width, 3), dtype = np.uint8)
y = np.zeros((n_train,), dtype = np.uint8)
half = int(n_train/2)

print('Loading train data.')
for i in tqdm(range(half)):
    X[i] = cv2.resize(cv2.imread(train_dir+'cat.'+str(i)+'.jpg'), (width, width))
    X[i+half] = cv2.resize(cv2.imread(train_dir+'dog.'+str(i)+'.jpg'), (width, width))
y[half:] = 1

def preprocess_input_ResNet50(x):
	return x - [103.939, 116.779, 123.68]

def get_features(MODEL, data = X):
	model = MODEL(include_top = False, input_shape = (width, width, 3), weights = 'imagenet')
	inputs = Input(shape = (width, width,3))
	x = inputs
	if MODEL == 'ResNet50' or MODEL == 'VGG16':
		x = Lambda(preprocess_input_ResNet50, name = 'preprocessing')(x)
	else: 
		x = Lambda(preprocess_input, name = 'preprocessing')(x)
	x = model(x)
	x = GlobalAveragePooling2D()(x)
	model = Model(inputs, x)
	
	features = model.predict(data, batch_size = 64, verbose = 1)
	return features

# Concatenate two models
print('Getting the features of Xception.')
Xception_features = get_features(Xception, X)
print('Getting the features of Inception V3.')
InceptionV3_features = get_features(InceptionV3, X)
features = np.concatenate([Xception_features, InceptionV3_features], axis = -1)

# Save the features
with h5py.File('features', 'w') as d:
	d['features'] = features
