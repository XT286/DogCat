# Data Processing Libraries
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import resnet50 as rn

# import from Keras Library
from keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D, Lambda
from keras.models import Model
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint


# Set constants
n_train = 25000
width = 224
train_dir = 'train/'

# Load image, resize to 224*224, normalize the three channels with the mean values
X = np.zeros((n_train, width, width, 3), dtype = np.uint8)
y = np.zeros((n_train,), dtype = np.uint8)
half = int(n_train/2)

for i in tqdm(range(half)):
    X[i] = cv2.resize(cv2.imread(train_dir+'cat.'+str(i)+'.jpg'), (width, width))
    X[i+half] = cv2.resize(cv2.imread(train_dir+'dog.'+str(i)+'.jpg'), (width, width))
y[half:] = 1

#Split validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

def preprocess_input(x):
	return x - [103.939, 116.779, 123.68]

image_input = Input(shape = (width, width,3))
image_input = Lambda(preprocess_input, name = 'preprocessing')(image_input)
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

callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='dog_cat.h', monitor='val_loss', save_best_only=True)]

customized_resnet_model.compile(loss='binary_crossentropy',optimizer='adam',
                            metrics=['accuracy'])


customized_resnet_model.fit(X_train, y_train, batch_size = 128, callbacks=callbacks, verbose=0, epochs = 20, validation_data = (X_val, y_val))

#customized_resnet_model.save('dog_cat.h')
