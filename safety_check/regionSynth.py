#!/usr/bin/env python

"""
compupute e_k according to e_{k-1} and p_{k-1}
author: Xiaowei Huang

"""

import numpy as np
import copy
from scipy import ndimage

import mnist
import cifar10
import imageNet
import mnist_network as NN_mnist
import cifar10_network as NN_cifar10
import imageNet_network as NN_imageNet

from conv_region_solve import conv_region_solve
from dense_region_solve import dense_region_solve
from regionByActivation import *
from regionByDerivative import *

from basics import *
from networkBasics import *
from configuration import * 

    
def regionSynth(model,dataset,image,manipulated,layer2Consider,span,numSpan,numDimsToMani):

    config = NN.getConfig(model)

    # get weights and bias of the entire trained neural network
    (wv,bv) = NN.getWeightVector(model,layer2Consider)
    
    # get the type of the current layer
    layerType = getLayerType(model,layer2Consider)

    wv2Consider, bv2Consider = getWeight(wv,bv,layer2Consider)
    
    # get the activations of the previous and the current layer
    if layer2Consider == 0: 
        activations0 = image
    else: activations0 = NN.getActivationValue(model,layer2Consider-1,image)
    activations1 = NN.getActivationValue(model,layer2Consider,image)

    if layerType == "Convolution2D":  
        print "convolutional layer, synthesising region ..."
        numDimsToMani = getManipulatedFeatureNumber(model,numDimsToMani,layer2Consider)
        if len(activations1.shape) == 3: 
            inds = getTop3D(model,image,activations1,manipulated,span.keys(),numDimsToMani,layer2Consider)
        elif len(activations1.shape) ==2: 
            inds = getTop2D(model,image,activations1,manipulated,span.keys(),numDimsToMani,layer2Consider)
        # filters can be seen as the output of a convolutional layer
        nfilters = numberOfFilters(wv2Consider)
        # features can be seen as the inputs for a convolutional layer
        nfeatures = numberOfFeatures(wv2Consider)
        (nextSpan,nextNumSpan) = conv_region_prep(model,dataBasics,nfeatures,nfilters,wv2Consider,bv2Consider,activations0,activations1,span,numSpan,inds,numDimsToMani)
    
    elif layerType == "Dense":
        print "dense layer, synthesising region ..."
        numDimsToMani = getManipulatedFeatureNumber(model,numDimsToMani,layer2Consider)
        inds = getTop(model,image,activations1,manipulated,numDimsToMani,layer2Consider)
        #print(inds)
        # filters can be seen as the output of a convolutional layer
        nfilters = numberOfFilters(wv2Consider)
        # features can be seen as the inputs for a convolutional layer
        nfeatures = numberOfFeatures(wv2Consider)
        (nextSpan,nextNumSpan) = dense_solve_prep(model,dataBasics,nfeatures,nfilters,wv2Consider,bv2Consider,activations0,activations1,span,numSpan,inds)
        
    elif layerType == "InputLayer":
        print "inputLayer layer, synthesising region ..."
        nextSpan = copy.deepcopy(span)
        nextNumSpan = copy.deepcopy(numSpan)
        
    elif layerType == "MaxPooling2D":
        print "MaxPooling2D layer, synthesising region ..."
        nextSpan = {}
        nextNumSpan = {}
        for key in span.keys():
            if len(key) == 3: 
                (k,i,j) = key
                i2 = i/2
                j2 = j/2
                nextSpan[k,i2,j2] = span[k,i,j]
                nextNumSpan[k,i2,j2] = numSpan[k,i,j]
            else: 
                print("error: ")
                    
    elif layerType == "Flatten":
        print "Flatten layer, synthesising region ..."
        nextSpan = copy.deepcopy(span)
        nextNumSpan = copy.deepcopy(numSpan)
        nextSpan = {}
        nextNumSpan = {}
        #print activations0[k-1][i-1][j-1]
        #print activations1[(k-1)*144+(i-1)*12+(j-1)]
        for key,value in span.iteritems():
            if len(key) == 3: 
                (k,i,j) = key
                il = len(activations0[0])
                jl = len(activations0[0][0])
                ind = k * il * jl + i * jl + jl
                nextSpan[ind] = span[key]
                nextNumSpan[ind] = numSpan[key]
    else: 
        print "Unknown layer type %s... "%(str(layerType))
        nextSpan = copy.deepcopy(span)
        nextNumSpan = copy.deepcopy(numSpan)
    return (nextSpan,nextNumSpan,numDimsToMani)
    

    
############################################################
#
#  preparation functions, from which to start SMT solving
#
################################################################
    
def conv_region_prep(model,dataBasics,nfeatures,nfilters,wv,bv,activations0,activations1,span,numSpan,inds,numDimsToMani):

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
    (nextSpan,nextNumSpan) = conv_region_solve(nfeatures,nfilters,filterCollection,biasCollection,activations0,activations1,span,numSpan,inds,numDimsToMani)
    print("found the region to work ")
    
    return (nextSpan,nextNumSpan)
    
def dense_solve_prep(model,dataBasics,nfeatures,nfilters,wv,bv,activations0,activations1,span,numSpan,inds):

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

    #for l in range(nfeatures):
    #    for k in range(nfilters):
    #        filter = [ w for ((p1,c1),(p,c),w) in wv if c1 == l+1 and c == k+1 ]
    #        bias = [ w for (p,c,w) in bv if c == k+1 ]
    #        if len(filter) == 0 or len(bias) == 0 : 
    #            print "error: bias =" + str(bias) + "\n filter = " + str(filter)
    #            break 
     #       else:
    #            filter = filter[0]
    #            bias = bias[0]
                    
    #        biasCollection[l,k] = bias
    #        filterCollection[l,k] = filter
            #print("%s,%s,%s,%s"%(l,k,bias,filter))
            
    (nextSpan,nextNumSpan) = dense_region_solve(nfeatures,nfilters,filterCollection,biasCollection,activations0,activations1,span,numSpan,inds)
    print("found the region to work ")
    #print nextSpan,nextNumSpan
    return (nextSpan,nextNumSpan)
    
    
############################################################
#
#  preparation functions, selecting heuristics
#
################################################################


def initialiseRegion(model,image,manipulated): 
    if heuristics == "Activation": 
        return initialiseRegionActivation(model,manipulated,image)
    elif heuristics == "Derivative": 
        return initialiseRegionDerivative(model,image,manipulated)
        
def getTop(model,image,activation,manipulated,numDimsToMani,layerToConsider): 
    if heuristics == "Activation": 
        return getTopActivation(activation,manipulated,layerToConsider,numDimsToMani)
    elif heuristics == "Derivative": 
        return getTopDerivative(model,image,activation,manipulated,numDimsToMani,layerToConsider)

def getTop2D(model,image,activation,manipulated,ps,numDimsToMani,layerToConsider): 
    if heuristics == "Activation": 
        return getTop2DActivation(activation,manipulated,ps,numDimsToMani,layerToConsider)
    elif heuristics == "Derivative": 
        return getTop2DDerivative(model,image,activation,manipulated,ps,numDimsToMani,layerToConsider)
        
def getTop3D(model,image,activation,manipulated,ps,numDimsToMani,layerToConsider): 
    if heuristics == "Activation": 
        return getTop3DActivation(activation,manipulated,ps,numDimsToMani,layerToConsider)
    elif heuristics == "Derivative": 
        return getTop3DDerivative(model,image,activation,manipulated,ps,numDimsToMani,layerToConsider)
