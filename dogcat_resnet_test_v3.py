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
from keras.callbacks import EarlyStopping, ModelCheckpoint


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

def get_features(MODEL = 'Xception', data = X):
	model = MODEL(include_top = False, input_shape = (width, width, 3), weights = 'imagenet')
	inputs = Input(shape = (width, width,3))
	x = inputs
	if MODEL == 'ResNet50':
		x = Lambda(preprocess_input_ResNet50, name = 'preprocessing')(x)
	else: 
		x = Lambda(preprocess_input, name = 'preprocessing')(x)
	x = model(x)
	x = GlobalAveragePooling2D()(x)
	model = Model(inputs, x)
	
	features = model.predict(data, batch_size = 64, verbose = 1)
	return features

# Read the features from File
with h5py.File('features', 'r') as d:
	features = np.array(d['features'])

# Build prediction model
inputs = Input(features.shape[1:])
x = inputs
x = Dropout(0.5)(x)
x = Dense(1, activation = 'sigmoid')(x)
final_model = Model(inputs, x)
final_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print('Training our final model.')
final_model.fit(features, y, batch_size = 128, epochs = 20, validation_split = 0.2)

# Construct test set
test_set = np.zeros((n_test, width, width, 3), dtype = np.uint8)

print('Loading the test data.')
for i in tqdm(range(n_test)):
    test_set[i] = cv2.resize(cv2.imread(test_dir+str(i+1)+'.jpg'), (width, width))

test_Xception_features = get_features(Xception, test_set)
test_InceptionV3_features = get_features(InceptionV3, test_set)
test_features = np.concatenate([test_Xception_features, test_InceptionV3_features], axis = -1)

print('Making Prediction, hang on.')
# predict using the final model
prediction = final_model.predict(test_features, batch_size = 128, verbose = 1)

# Output labels
df = pd.read_csv('sample_submission.csv')
df['label'] = prediction.clip(min = 0.005, max = 0.995)
df.to_csv('prediction.csv', index = None)

