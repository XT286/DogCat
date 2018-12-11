# Data Processing Libraries
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import cv2
import random
from sklearn.model_selection import train_test_split

# import from Keras Library
from keras.layers import Input, Conv2D, BatchNormalization, Activation, ZeroPadding2D, Dense, Flatten, Dropout 
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D,AveragePooling2D
from keras import layers
import keras.backend as K
from keras.utils import layer_utils
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.models import Model

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

# Define identity block in ResNet. 
# Code borrowed from https://www.kaggle.com/ryanmarfty/dogcat-res50
def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

# Define convolution block in ResNet. 
# Code borrowed from https://www.kaggle.com/ryanmarfty/dogcat-res50
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

# Define ResNet. 
# Code borrowed from https://www.kaggle.com/ryanmarfty/dogcat-res50
def ResNet50(include_top=True, weights=None,input_tensor=None, input_shape=None,pooling=None,classes=1000):
    input_shape = _obtain_input_shape(input_shape,default_size=224,min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes,kernel_initializer='lecun_normal' ,activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = 'ResNet50_Pretrained/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        else:
            weights_path = 'ResNet50_Pretrained/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')
    return model

image_input = Input(shape = (width, width,3))
model = ResNet50(input_tensor = image_input, include_top = False, weights = 'imagenet')

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
