#!/usr/bin/env python

"""
compupute e_k according to e_{k-1} and p_{k-1}
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

    
def regionSynth(model,dataset,image,manipulated,layer2Consider,cl,gl,pk,mfn):

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
        print "convolutional layer, synthesising region ..."
        mfn = getManipulatedFeatureNumber(model,mfn,layer2Consider)
        if len(activations1.shape) == 3: 
            inds = getTop3D(model,image,activations1,manipulated,cl.keys(),mfn,layer2Consider)
        elif len(activations1.shape) ==2: 
            inds = getTop2D(model,image,activations1,manipulated,cl.keys(),mfn,layer2Consider)
        # filters can be seen as the output of a convolutional layer
        nfilters = numberOfFilters(wv2Consider)
        # features can be seen as the inputs for a convolutional layer
        nfeatures = numberOfFeatures(wv2Consider)
        (ncl,ngl) = conv_solve_prep(model,dataBasics,nfeatures,nfilters,wv2Consider,bv2Consider,activations0,activations1,cl,gl,inds,mfn)
    
    elif layerType == "Dense":
        print "dense layer, synthesising region ..."
        mfn = getManipulatedFeatureNumber(model,mfn,layer2Consider)
        inds = getTop(model,image,activations1,manipulated,mfn,layer2Consider)
        #print(inds)
        # filters can be seen as the output of a convolutional layer
        nfilters = numberOfFilters(wv2Consider)
        # features can be seen as the inputs for a convolutional layer
        nfeatures = numberOfFeatures(wv2Consider)
        (ncl,ngl) = dense_solve_prep(model,dataBasics,nfeatures,nfilters,wv2Consider,bv2Consider,activations0,activations1,cl,gl,inds)
        
    elif layerType == "InputLayer":
        print "inputLayer layer, synthesising region ..."
        ncl = copy.deepcopy(cl)
        ngl = copy.deepcopy(gl)
        
    elif layerType == "MaxPooling2D":
        print "MaxPooling2D layer, synthesising region ..."
        ncl = {}
        ngl = {}
        for key in cl.keys():
            if len(key) == 3: 
                (k,i,j) = key
                i2 = i/2
                j2 = j/2
                ncl[k,i2,j2] = cl[k,i,j]
                ngl[k,i2,j2] = gl[k,i,j]
            else: 
                print("error: ")
                    
    elif layerType == "Flatten":
        print "Flatten layer, synthesising region ..."
        ncl = copy.deepcopy(cl)
        ngl = copy.deepcopy(gl)
        ncl = {}
        ngl = {}
        #print activations0[k-1][i-1][j-1]
        #print activations1[(k-1)*144+(i-1)*12+(j-1)]
        for key,value in cl.iteritems():
            if len(key) == 3: 
                (k,i,j) = key
                il = len(activations0[0])
                jl = len(activations0[0][0])
                ind = k * il * jl + i * jl + jl
                ncl[ind] = cl[key]
                ngl[ind] = gl[key]
    else: 
        print "Unknown layer type %s... "%(str(layerType))
        ncl = copy.deepcopy(cl)
        ngl = copy.deepcopy(gl)
    return (ncl,ngl,mfn)
    

    
############################################################
#
#  preparation functions, from which to start SMT solving
#
################################################################
    
def conv_solve_prep(model,dataBasics,nfeatures,nfilters,wv,bv,activations0,activations1,cl,gl,inds,mfn):

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
    (ncl,ngl) = conv_region_solve(nfeatures,nfilters,filterCollection,biasCollection,activations0,activations1,cl,gl,inds,mfn)
    print("found the region to work ")
    
    return (ncl,ngl)
    
def dense_solve_prep(model,dataBasics,nfeatures,nfilters,wv,bv,activations0,activations1,cl,gl,inds):

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
            
    (ncl,ngl) = dense_region_solve(nfeatures,nfilters,filterCollection,biasCollection,activations0,activations1,cl,gl,inds)
    print("found the region to work ")
    #print ncl,ngl
    return (ncl,ngl)
    
    
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
        
def getTop(model,image,activation,manipulated,mfn,layerToConsider): 
    if heuristics == "Activation": 
        return getTopActivation(activation,manipulated,layerToConsider,mfn)
    elif heuristics == "Derivative": 
        return getTopDerivative(model,image,activation,manipulated,mfn,layerToConsider)

def getTop2D(model,image,activation,manipulated,ps,mfn,layerToConsider): 
    if heuristics == "Activation": 
        return getTop2DActivation(activation,manipulated,ps,mfn,layerToConsider)
    elif heuristics == "Derivative": 
        return getTop2DDerivative(model,image,activation,manipulated,ps,mfn,layerToConsider)
        
def getTop3D(model,image,activation,manipulated,ps,mfn,layerToConsider): 
    if heuristics == "Activation": 
        return getTop3DActivation(activation,manipulated,ps,mfn,layerToConsider)
    elif heuristics == "Derivative": 
        return getTop3DDerivative(model,image,activation,manipulated,ps,mfn,layerToConsider)
