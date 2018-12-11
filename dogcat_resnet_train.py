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
from keras.preprocessing import image

# Load data, first getting the names of the photos
train_dir = 'train/'
width = 224

train_images = [train_dir+name for name in os.listdir(train_dir)]

# Load image, resize to 224*224, normalize the three channels with the mean values
def read_image(image):
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (width,width), interpolation = cv2.INTER_CUBIC)
    img = img - np.array([123.68, 116.779, 103.939]).reshape(1,1,3)
    return img

# Prep the data by getting the labels (dog or cat) from the file names.
def prep_images(image_dir):
    labels = []
    count = len(image_dir)
    data = np.ndarray((count, width, width, 3), dtype = np.float32)
    
    for i, image in enumerate(image_dir):
        image = read_image(image)
        data[i] = image
        true_label = image[6:9]
        if true_label == 'dog': labels.append(1)
        else: labels.append(0)
    return data, labels

# We randomly pick our training and validation set
validationset = random.sample(train_images, 2000)
trainset = list(set(train_images)-set(validationset))
train, train_labels = prep_images(trainset)
validation, val_labels= prep_images(validationset)

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

# We only train the last layers
for layer in customized_resnet_model.layers[:-5]:
    layer.trainable = False

datagen = image.ImageDataGenerator()
train_generator = datagen.flow(train,train_labels,batch_size=32)
val_generator = datagen.flow(validation,val_labels,batch_size=16)

customized_resnet_model.compile(loss='binary_crossentropy',optimizer='RMSprop',
                            metrics=['accuracy'])
customized_resnet_model.fit_generator(train_generator, steps_per_epoch=50,
                                  validation_data=val_generator, validation_steps=20,
                                  epochs=5, verbose=1)

customized_resnet_model.save('dog_cat.h')
