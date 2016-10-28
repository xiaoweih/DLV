#!/usr/bin/env python

"""
compupute p_k according to e_k = (nextSpan,nextNumSpan), e_{k-1} = (cl,gl) and p_{k-1}
author: Xiaowei Huang
"""


import numpy as np
import copy 
from scipy import ndimage
import math

from conv_precision_solve import conv_precision_solve
from dense_precision_solve import dense_precision_solve

from networkBasics import * 
from configuration import * 
from basics import *

"""
The following is an overapproximation
"""

def precisionSynth(layer2Consider,nextSpan,nextNumSpan):

    if layer2Consider in errorBounds.keys(): 
        pk = errorBounds[layer2Consider]
    else: 
        pk = errorBounds[-1]
    
    for k in nextSpan.keys(): 
        length = nextSpan[k] * nextNumSpan[k]
        nextNumSpan[k] = math.ceil(length / float(pk))
        nextSpan[k] = pk
        
    return (nextSpan,nextNumSpan,pk)


"""

def precisionSynth(model,dataset,image,layer2Consider,cl,gl,nextSpan,nextNumSpan,pk):


    config = NN.getConfig(model)

    # get weights and bias of the entire trained neural network
    (wv,bv) = NN.getWeightVector(model,layer2Consider)
    
    # get the activations of the previous and the current layer
    if layer2Consider == 0: 
        activations0 = image
    else: activations0 = NN.getActivationValue(model,layer2Consider-1,image)
    activations1 = NN.getActivationValue(model,layer2Consider,image)

    # get the type of the current layer
    layerType = getLayerType(model,layer2Consider)
    #[ lt for (l,lt) in config if layer2Consider == l ]
    #if len(layerType) > 0: layerType = layerType[0]
    #else: print "cannot find the layerType"

    wv2Consider, bv2Consider = getWeight(wv,bv,layer2Consider)

    if layerType == "Convolution2D":  
        print "convolutional layer, synthesising precision ..."
        # filters can be seen as the output of a convolutional layer
        nfilters = numberOfFilters(wv2Consider)
        # features can be seen as the inputs for a convolutional layer
        nfeatures = numberOfFeatures(wv2Consider)
        npk = conv_solve_prep(model,dataBasics,nfeatures,nfilters,wv2Consider,bv2Consider,activations0,activations1,cl,gl,nextSpan,nextNumSpan,pk)
        
    elif layerType == "Dense":  
        print "dense layer, synthesising precision ..."
        # filters can be seen as the output of a convolutional layer
        nfilters = numberOfFilters(wv2Consider)
        # features can be seen as the inputs for a convolutional layer
        nfeatures = numberOfFeatures(wv2Consider)
        npk = dense_solve_prep(model,dataBasics,nfeatures,nfilters,wv2Consider,bv2Consider,activations0,activations1,cl,gl,nextSpan,nextNumSpan,pk)
        
    elif layerType == "InputLayer":  
        print "inputLayer layer, synthesising precision ..."
        npk = copy.copy(pk)
    else: 
        npk = copy.copy(pk)

    return npk
    
    
def conv_solve_prep(model,dataBasics,nfeatures,nfilters,wv,bv,activations0,activations1,cl,gl,nextSpan,nextNumSpan,pk):

    # space holders for computation values
    biasCollection = {}
    filterCollection = {}
    
    for l in range(nfeatures):
        for k in range(nfilters):
            filter = [ w for ((p1,c1),(p,c),w) in wv if c1 == l+1 and c == k+1 ]
            bias = [ w for (p,c,w) in bv if c == k+1 ]
            if len(filter) == 0 or len(bias) == 0 : 
                print "error: bias =" + str(bias) + "\n filter = " + str(filter)
            else:
                filter = filter[0]
                bias = bias[0]
                    
            # flip the filter for convolve
            flipedFilter = np.fliplr(np.flipud(filter))
            biasCollection[l,k] = bias
            filterCollection[l,k] = flipedFilter
            #print filter.shape
            
    npk = conv_precision_solve(nfeatures,nfilters,filterCollection,biasCollection,activations0,activations1,cl,gl,nextSpan,nextNumSpan,pk)
    #print("found the region to work ")
    
    return npk
    
    
def dense_solve_prep(model,dataBasics,nfeatures,nfilters,wv,bv,activations0,activations1,cl,gl,nextSpan,nextNumSpan,pk):

    # space holders for computation values
    biasCollection = {}
    filterCollection = {}
  
    for ((p1,c1),(p,c),w) in wv: 
        if c1-1 in range(nfeatures) and c-1 in range(nfilters): 
            filterCollection[c1-1,c-1] = w
    for (p,c,w) in bv: 
        if c-1 in range(nfilters): 
            for l in range(nfeatures): 
                biasCollection[l,c-1] = w
            
    npk = dense_precision_solve(nfeatures,nfilters,filterCollection,biasCollection,activations0,activations1,cl,gl,nextSpan,nextNumSpan,pk)
    #print("found the region to work ")
    
    return npk
    
"""
