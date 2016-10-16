#!/usr/bin/env python

from __future__ import print_function

import scipy.io as sio
import numpy as np
import copy

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils



# for cifar10
from keras.datasets import cifar10
from keras.optimizers import SGD
#

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

def read_dataset():

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    return (X_train,Y_train,X_test,Y_test, img_channels, img_rows, img_cols, batch_size, nb_classes, nb_epoch, data_augmentation)

def build_model(img_channels, img_rows, img_cols, nb_classes):

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    return model
    
def read_model_from_file(img_channels, img_rows, img_cols, nb_classes, weightFile,modelFile):
    """
    define neural network model
    :return: network model
    """
    
    model = build_model(img_channels, img_rows, img_cols, nb_classes)
    model.summary()
    
    weights = sio.loadmat(weightFile)
    model = model_from_json(open(modelFile).read())
    for (idx,lvl) in [(1,0),(2,2),(3,6),(4,8),(5,13),(6,16)]:
        
        weight_1 = 2 * idx - 2
        weight_2 = 2 * idx - 1
        model.layers[lvl].set_weights([weights['weights'][0, weight_1], weights['weights'][0, weight_2].flatten()])

    return model

"""
   The following function gets the activations for a particular layer
   for an image in the test set. 
   FIXME: ideally I would like to be able to 
          get activations for a particular layer from the inputs of another layer. 
"""

    
def getImage(model,n_in_tests):

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.astype('float32')
    X_test = X_test.astype('float32')

    
    X_test /= 255
    
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    image = X_test[n_in_tests:n_in_tests+1]
    
    #print(np.amax(image),np.amin(image))
        
    return np.squeeze(image)
    
def readImage(path):

    import cv2
    
    im = cv2.resize(cv2.imread(path), (img_rows, img_cols)).astype('float32')
    im = im / 255
    im = im.transpose(2, 0, 1)
    
    #print(np.amax(im),np.amin(im))

    
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

    newInput_for_predict = copy.deepcopy(newInput)
    newInput2 = np.expand_dims(newInput_for_predict, axis=0)
    predictValue = model.predict(newInput2)
    newClass = np.argmax(np.ravel(predictValue))
    confident = np.amax(np.ravel(predictValue))
    return (newClass,confident)    
    
def getWeightVector(model, layer2Consider):
    weightVector = []
    biasVector = []

    for layer in model.layers:
    	 index=model.layers.index(layer)
         h=layer.get_weights()
         
         if len(h) > 0 and index in [0,2,6,8]  and index <= layer2Consider : 
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
                     
         elif len(h) > 0 and index in [13,16]  and index <= layer2Consider: 
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