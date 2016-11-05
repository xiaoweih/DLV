#!/usr/bin/env python

"""
author: Xiaowei Huang
"""

import numpy as np
import scipy.io as sio

# keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense
import keras.optimizers
from keras.models import model_from_json
from keras import backend as K

# visualisation
from keras.utils.visualize_util import plot
#
import basics 

from twoDcurve import target_fun

TWO_PI = 2 * np.pi



def load_data(N_samples,N_tests):
    """
    N_samples: number of samples
    """
    # manipulated data
    np.random.seed(basics.current_milli_time())
    # use the following if need to manipulate the samples
    #x1 = manipulateSample(np.random.random((N_samples, 1)) * TWO_PI)
    x1 = np.random.random((N_samples, 1)) * TWO_PI
    x2 = np.random.random((N_samples, 1)) * TWO_PI
    #x3 = np.random.random((N_samples, 1)) * TWO_PI

    x0 = constructInputOutputArray2([x1,x2])

    c = [(e, mapping(e)) for e in x0] 
    (x,y) = zip(*c)
    
    x = np.array(x0)
    y = np.array(y) 
    
    # un-manipulated data
    xx1 = np.random.random((N_tests, 1)) * TWO_PI
    xx2 = np.random.random((N_tests, 1)) * TWO_PI
    #x3 = np.random.random((N_samples, 1)) * TWO_PI
    xx0 = constructInputOutputArray2([xx1,xx2])

    xc = [(e, mapping(e)) for e in xx0] 
    (xx,xy) = zip(*xc)

    xx = np.array(xx0)
    xy = np.array(xy) 

    I = np.arange(N_samples)
    np.random.shuffle(I)

    #return x_train, y_train, x_test, y_test
    return x, y, xx, xy

# get a random 
def getImage(model,n_in_tests):

    N_samples = 1
    N_tests = 1
    x_train, y_train, x_test, y_test = load_data(N_samples,N_tests)
    return x_train[0]
    
def mapping(x):
    p = target_fun(x)
    y1 = x[1] >= p
    y2 = x[1] < p
    return [y1,y2]

def read_model_from_file(weightFile,modelFile):
    """
    define neural network model
    :return: network model
    """
    model = build_model()
    model.summary()

    weights = sio.loadmat(weightFile)
    model = model_from_json(open(modelFile).read())
    for idx in range(1, 4):
        weight_1 = 2 * idx - 2
        weight_2 = 2 * idx - 1
        model.layers[idx].set_weights([weights['weights'][0, weight_1], weights['weights'][0, weight_2].flatten()])

    return model
    
def getActivationValue(model,layer,image):
    image = np.expand_dims(image, axis=0)
    activations = get_activations(model, layer, image)
    return np.squeeze(activations)
    
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations
    

def getConfig(model):

    config = model.get_config()
    config = [ getLayerName(dict) for dict in config.get('layers') ]
    config = zip(range(len(config)),config)
    return config 
    
def getLayerName(dict):

    className = dict.get('class_name')
    if className == 'Activation': 
        return dict.get('config').get('activation')
    else: 
        return className

def build_model(batch_num=None):
    """
    define neural network model
    """

    data = Input(batch_shape=(batch_num, 2), name='data')

    fc_1 = Dense(20, activation='relu', init='normal', name='fc_1')(data)
    fc_2 = Dense(20, activation='relu', init='normal', name='fc_2')(fc_1)
    #fc_3 = Dense(20, activation='relu', init='normal', name='fc_3')(fc_2)
    output = Dense(2, init='normal', name='output')(fc_2)

    # define model
    model = Model(input=[data], output=[output])

    return model

def manipulateSample(x): 
    y = []
    for k in range(len(x)):
        if inRealRange(x[k][0],1,1.5) or inRealRange(x[k][0],3,3.5): y.append([x[k][0]-0.5])
        elif inRealRange(x[k][0],1.5,2) or inRealRange(x[k][0],3.5,4): y.append([x[k][0] + 0.5])
        else: y.append([x[k][0]])
    return np.array(y)
    
def inRealRange(x,c1,c2): 
    return x >= c1 and x <= c2

def constructInputOutputArray3(x):
    return [ (zip(x1,x2,x3))[0] for (x1,x2,x3) in zip(x[0],x[1],x[2])]
    
def constructInputOutputArray2(x):
    return [ (zip(x1,x2))[0] for (x1,x2) in zip(x[0],x[1]) ]
    
def predictWithImage(model,newInput):
    newInput2 = np.expand_dims(newInput, axis=0)
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
         
         if len(h) > 0 and index in [1,2,3]  and index <= layer2Consider: 
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