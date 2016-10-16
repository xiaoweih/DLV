#!/usr/bin/env python

from __future__ import print_function

import scipy.io as sio
import numpy as np
import os

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
from keras.utils import np_utils


#
from keras.optimizers import SGD

directory_model_string = "networks"

batch_size = 32
nb_classes = 1000
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 224, 224
# the CIFAR10 images are RGB
img_channels = 3

def read_dataset():

    #FIXME: at the moment, we dont know where to read dataset directly 
    return (img_channels, img_rows, img_cols, batch_size, nb_classes, nb_epoch, data_augmentation)

def build_model(img_channels, img_rows, img_cols, nb_classes):

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(img_channels, img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')



    return model
    
def read_model_from_file(img_channels, img_rows, img_cols, nb_classes, weightFile,modelFile):
    """
    define neural network model
    :return: network model
    """
    
    #model = build_model(img_channels, img_rows, img_cols, nb_classes)

    
    model = model_from_json(open(modelFile).read())
    model.summary()
    
    weights = sio.loadmat(weightFile)
        
    for (idx,lvl) in [(1,1),(2,3),(3,6),(4,8),(5,11),(6,13),(7,15),(8,18),(9,20),(10,22),(11,25),(12,27),(13,29),(14,32),(15,34),(16,36)]:
        
        weight_1 = 2 * idx - 2
        weight_2 = 2 * idx - 1
        model.layers[lvl].set_weights([weights['weights'][0, weight_1], weights['weights'][0, weight_2].flatten()])

    return model

"""
    if weights_path:
        model.load_weights(weights_path)

im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = VGG_16('vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    print np.argmax(out)
"""

def removeZeroPadding2D(image):

    nb_channels = len(image)
    rows = len(image[0])
    cols = len(image[0][0])
    
    return image[:,1:rows-1,1:cols-1]
    
def addZeroPadding2D(image):

    nb_channels = len(image)
    rows = len(image[0])
    cols = len(image[0][0])
    
    image0 = np.zeros((nb_channels,rows+2,cols+2))
    image0[:,1:rows+1,1:cols+1] = image
    
    return image0
    
def getImage(model,n_in_tests):

    import cv2
    
    directory = "%s/imageNet/ImageNet_Utils/n02084071/n02084071_urlimages/"%directory_model_string
    traffic = "%s/imageNet/traffic_signs/"%directory_model_string
    street = "%s/imageNet/street_sign/"%directory_model_string

    working_directory =   street # traffic #  directory #
    allfiles = [file for file in os.listdir(working_directory) if file.endswith(".jpg") ] 
    im = cv2.resize(cv2.imread(working_directory+allfiles[n_in_tests]), (224, 224)).astype(np.float32)
    
    #print((im.transpose(2,0,1))[0][1])
    
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    
    #print(np.amax(im),np.amin(im))

    im = np.expand_dims(im, axis=0)
    
    return np.squeeze(im)
    
    
def readImage(path):

    import cv2
    
    im = cv2.resize(cv2.imread(path), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    
    return np.squeeze(im)


def getActivationValue(model,layer,image):

    image = np.expand_dims(image, axis=0)
    activations = get_activations(model, layer, image)
    return np.squeeze(activations)

    
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations
    
def predictWithImage(model,newInput):

    newInput2 = np.expand_dims(newInput, axis=0)
    predictValue = model.predict(newInput2)
    newClass = np.argmax(np.ravel(predictValue))
    confident = np.amax(np.ravel(predictValue))
    return (newClass,confident)    
    
def getWeightVector(model,layer2Consider):
    weightVector = []
    biasVector = []

    for layer in model.layers:
    	 index=model.layers.index(layer)
         h=layer.get_weights()
         
         if len(h) > 0 and index in [1,3,6,8,11,13,15,18,20,22,25,27,29] and index <= layer2Consider: 
         # for convolutional layer
             ws = h[0]
             bs = h[1]
             
             #print("layer =" + str(index))
             #print(layer.input_shape)
             #print(ws.shape)
             #print(bs.shape)

             
             # number of filters in the previous layer
             m = len(ws)
             # number of features in the previous layer
             # every feature is represented as a matrix 
             n = len(ws[0])
             
             for i in range(1,m+1):
                 
                 biasVector.append((index,i,h[1][i-1]))
             
             for i in range(1,m+1):
                 v = ws[i-1]
                 for j in range(1,n+1): 
                     # (feature, filter, matrix)
                     weightVector.append(((index,j),(index,i),v[j-1]))
                     
         elif len(h) > 0 and index in [32,34,36] and index <= layer2Consider: 
         # for fully-connected layer
             ws = h[0]
             bs = h[1]
             
             # number of nodes in the previous layer
             m = len(ws)
             # number of nodes in the current layer
             n = len(ws[0])
             
             for j in range(1,n+1):
                 biasVector.append((index,j,h[1][j-1]))
             
             for i in range(1,m+1):
                 v = ws[i-1]
                 for j in range(1,n+1): 
                     weightVector.append(((index-1,i),(index,j),v[j-1]))
         #else: print "\n"
         
    return (weightVector,biasVector)        
    
def getConfig(model):

    config = model.get_config()
    config = [ getLayerName(dict) for dict in config ]
    config = zip(range(len(config)),config)
    return config 
    
def getLayerName(dict):

    className = dict.get('class_name')
    if className == 'Activation': 
        return dict.get('config').get('activation')
    else: 
        return className