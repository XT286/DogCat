# Data Processing Libraries
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import cv2
import random
from sklearn.model_selection import train_test_split
import resnet50 as rn

# import from Keras Library
from keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Model


# Load data, first getting the names of the photos
test_dir = 'test/'
n_test = 12500
width = 224


image_input = Input(shape = (width, width,3))
model = rn.ResNet50(input_tensor = image_input, include_top = False, weights = 'imagenet')

# Define the last layers for our customized problem
def customized_output(model):
    last_layer = model.output
    x = GlobalAveragePooling2D()(last_layer)
    x = Dropout(0.1)(x)
    x = Dense(256, kernel_initializer = 'lecun_normal', 
              activation = 'relu', name = 'fc2')(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation = 'sigmoid', name = 'output')(x)
    return out

output = customized_output(model)
customized_resnet_model = Model(inputs = model.input, outputs = output)

# Load the pretrained weights for the whole model
for layer in customized_resnet_model.layers:
    layer.trainable = False
dog_cat_weightpath = 'dog_cat.h'
customized_resnet_model.load_weights(dog_cat_weightpath)

test_set = np.zeros((n_test, width, width, 3), dtype = np.uint8)

for i in tqdm(range(n_test)):
    test_set[i] = cv2.resize(cv2.imread(test_dir+str(i+1)+'.jpg'), (width, width))

# predict using the customized resnet model
prediction = customized_resnet_model.predict(test_set, batch_size = 128, verbose = 1)

# Output labels
df = pd.read_csv('sample_submission.csv')
df['label'] = prediction.clip(min = 0.005, max = 0.995)
df.to_csv('prediction.csv', index = None)
