#!/usr/bin/env python

"""
compupute e_k according to e_{k-1} and p_{k-1}
author: Xiaowei Huang

"""

import numpy as np
import copy
from scipy import ndimage


from basics import *
from configuration import * 
from networkBasics import * 

    
############################################################
#
#  initialise a region for the input 
#
################################################################   

 
def initialiseRegionDerivative(model,image,manipulated): 


    # get the type of the current layer
    layerType = getLayerType(model,0)
    #[ lt for (l,lt) in config if l == 0 ]
    #if len(layerType) > 0: layerType = layerType[0]
    #else: print "cannot find the layerType"
    
    if layerType == "Convolution2D":
        nextSpan = {}
        nextNumSpan = {}
        if len(image.shape) == 2: 
            # decide how many elements in the input will be considered
            if len(image)*len(image[0])  < featureDims : 
                numDimsToMani = len(image)*len(image[0])
            else: numDimsToMani = featureDims
            # get those elements with maximal/minimum values
            ls = getTop2DDerivative(model,image,image,manipulated,numDimsToMani,-1)
        elif len(image.shape) == 3:
            # decide how many elements in the input will be considered
            if len(image)*len(image[0])*len(image[0][0])  < featureDims : 
                numDimsToMani = len(image)*len(image[0])*len(image[0][0])
            else: numDimsToMani = featureDims
            # get those elements with maximal/minimum values
            ls = getTop3DDerivative(model,image,image,manipulated,numDimsToMani,-1)         
        for i in ls: 
            nextSpan[i] = span
            nextNumSpan[i] = numSpan

    elif layerType == "InputLayer":
        nextSpan = {}
        nextNumSpan = {}
        # decide how many elements in the input will be considered
        if len(image)  < featureDims : 
            numDimsToMani = len(image) 
        else: numDimsToMani = featureDims
        # get those elements with maximal/minimum values
        ls = getTopDerivative(model,image,image,manipulated,numDimsToMani,-1)
        for i in ls: 
            nextSpan[i] = span
            nextNumSpan[i] = numSpan
            
    elif layerType == "ZeroPadding2D": 
        #image1 = addZeroPadding2D(image)
        image1 = image
        nextSpan = {}
        nextNumSpan = {}
        if len(image1.shape) == 2: 
            # decide how many elements in the input will be considered
            if len(image1)*len(image1[0])  < featureDims : 
                numDimsToMani = len(image1)*len(image1[0]) 
            else: numDimsToMani = featureDims
            # get those elements with maximal/minimum values
            ls = getTop2DDerivative(model,image,image,manipulated,numDimsToMani,-1)

        elif len(image1.shape) == 3:
            # decide how many elements in the input will be considered
            if len(image1)*len(image1[0])*len(image1[0][0])  < featureDims : 
                numDimsToMani = len(image1)*len(image1[0])*len(image1[0][0])
            else: numDimsToMani = featureDims
            # get those elements with maximal/minimum values
            ls = getTop3DDerivative(model,image,image,manipulated,numDimsToMani,-1)         
        for i in ls: 
            nextSpan[i] = span
            nextNumSpan[i] = numSpan
            
    else: 
        print "initialiseRegionDerivative: Unknown layer type ... "
        
    return (nextSpan,nextNumSpan,numDimsToMani)
    
    

    
############################################################
#
#  auxiliary functions
#
################################################################
    


def getTopDerivative(model,image0,image,manipulated,numDimsToMani,layerToConsider): 

    avoid = repeatedManipulation == "disallowed" and layerToConsider == -1
    
    avg = np.sum(image)/float(len(image))
    nimage = derivative(model,image0,manipulated,layerToConsider,derivativelayerUpTo)

    topImage = {}
    toBeDeleted = []
    for i in range(len(image)):
        if len(topImage) < numDimsToMani: 
            topImage[i] = nimage[i]
        else: 
            bl = False
            for k, v in topImage.iteritems():
                if v < nimage[i] and not (k in toBeDeleted) and ((not avoid) or (i not in manipulated)): 
                        toBeDeleted.append(k)
                        bl = True
                        break
            if bl == True: 
                topImage[i] = nimage[i]
    for k in toBeDeleted: 
        del topImage[k]
    return topImage.keys()
    
def getTop2DDerivative(model,image0,image,manipulated,numDimsToMani,layerToConsider): 

    avoid = repeatedManipulation == "disallowed" and layerToConsider == -1

    avg = np.sum(image)/float(len(image)*len(image[0]))
    nimage = derivative(model,image0,manipulated,-1,derivativelayerUpTo)

    topImage = {}
    toBeDeleted = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            if len(topImage) < numDimsToMani: 
                topImage[(i,j)] = nimage[(i,j)]
            else: 
                bl = False 
                for (k1,k2), v in topImage.iteritems():
                    if v < nimage[(i,j)] and not ((k1,k2) in toBeDeleted) and ((not avoid) or ((i,j) not in manipulated)):  
                        toBeDeleted.append((k1,k2))
                        bl = True
                        break
                if bl == True: 
                    topImage[(i,j)] = nimage[(i,j)]
    for (k1,k2) in toBeDeleted: 
        del topImage[(k1,k2)]
    return topImage.keys()
    

##################
# get top elements that are connected to those of previous layer
##################

# ps are indices of the previous layer

def getTop3DDerivative(model,image0,image,manipulated,ps,numDimsToMani,layerToConsider): 

    avoid = repeatedManipulation == "disallowed" and layerToConsider == -1

    avg = np.sum(image)/float(len(image)*len(image[0]*len(image[0][0])))
    nimage = derivative(model,image0,manipulated,layerToConsider,derivativelayerUpTo)
                
    # do not care about the first dimension
    # only care about individual convolutional node
    if len(ps[0]) == 3: 
        (p1,p2,p3) = zip(*ps)
        ps = zip(p2,p3)
    ks = []
    pointsToConsider = []
    for i in range(numDimsToMani): 
        if i <= len(ps) -1: 
            (x,y) = ps[i] 
            nps = [ (x-x1,y-y1) for x1 in range(filterSize) for y1 in range(filterSize) if x-x1 >= 0 and y-y1 >=0 ]
            pointsToConsider = pointsToConsider + nps
            ks = ks + findFromArea3D(image,manipulated,avoid,nimage,nps,1,ks)

        else: 
            ks = ks + findFromArea3D(image,manipulated,avoid,nimage,pointsToConsider,1,ks)
    return ks
    
def findFromArea3D(image,manipulated,avoid,nimage,ps,numDimsToMani,ks):
    topImage = {}
    toBeDeleted = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(len(image[0][0])):
                if len(topImage) < numDimsToMani and ((j,k) in ps or len(ps) == 0) and (i,j,k) not in ks: 
                    topImage[(i,j,k)] = nimage[(i,j,k)]
                elif ((j,k) in ps or len(ps) == 0) and (i,j,k) not in ks: 
                    bl = False 
                    for (k1,k2,k3), v in topImage.iteritems():
                        if v < nimage[(i,j,k)] and not ((k1,k2,k3) in toBeDeleted) and ((not avoid) or ((i,j,k) not in manipulated)):  
                            toBeDeleted.append((k1,k2,k3))
                            bl = True
                            break
                    if bl == True: 
                        topImage[(i,j,k)] = nimage[(i,j,k)]
    for (k1,k2,k3) in toBeDeleted: 
        del topImage[(k1,k2,k3)]
    return topImage.keys()


##################
# get Derivative for a neuron
# layer = -1 means input layer
# layer is the layer for the current point
# layer1 = layer + 1 is the layer for the next 
##################

def derivative(model,image,manipulated,layer,layerUpTo):

    # get the type of the current layer
    layer1 = layer + 1
    
    if layer >= 0: 
        activations0 = NN.getActivationValue(model,layer,image)    
    else: activations0 = image
    activations1 = NN.getActivationValue(model,layer1,image)

    # initialise derivatives
    derivatives = copy.deepcopy(activations0)

    if layer == layerUpTo: 
        
        if len(activations0.shape) == 3: 
            for k in range(len(activations0)): 
                for x in range(len(activations0[0])): 
                    for y in range(len(activations0[0][0])): 
                        derivatives[k][x][y] = 1.0
        elif len(activations0.shape) == 2: 
            for x in range(len(activations0[0])): 
                for y in range(len(activations0[0][0])): 
                    derivatives[x][y] = 1.0
        return derivatives
    else: 
        value1 = derivative(model,image,manipulated,layer1,layerUpTo)
        layerType = getLayerType(model,layer1)
        #[ lt for (l,lt) in config if l == layer1 ]
        #layerType = layerType[0]
        (wv,bv) = NN.getWeightVector(model,layer1)
        wv1, bv1 = getWeight(wv,bv,layer1)

        if len(wv1) > 0 and layerType == "Convolution2D": 
            # filters can be seen as the output of a convolutional layer
            nfilters = numberOfFilters(wv1)
            # features can be seen as the inputs for a convolutional layer
            nfeatures = numberOfFeatures(wv1)
            # space holders for computation values
            biasCollection = {}
            filterCollection = {}
    
            for l in range(nfeatures):
                for k in range(nfilters):
                    filter = [ w for ((p1,c1),(p2,c),w) in wv1 if c1 == l+1 and c == k+1 ]
                    bias = [ w for (p2,c,w) in bv1 if c == k+1 ]
                    if len(filter) == 0 or len(bias) == 0 : 
                        print "error: bias =" + str(bias) + "\n filter = " + str(filter)
                    else:
                        filter = filter[0]
                        bias = bias[0]
                    # flip the filter for convolve
                    flipedFilter = np.fliplr(np.flipud(filter))
                    biasCollection[l,k] = bias
                    filterCollection[l,k] = flipedFilter
        elif len(wv1) > 0 and layerType == "Dense" : 
            # filters can be seen as the output of a convolutional layer
            nfilters = numberOfFilters(wv1)
            # features can be seen as the inputs for a convolutional layer
            nfeatures = numberOfFeatures(wv1)
            
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
                    
            biasCollection[l,k] = bias
            filterCollection[l,k] = filter
        
        if layerType == "relu" :
            if len(activations1.shape) == 2: 
                for x in range(len(activations1)):
                    for y in range(len(activations1[0])): 
                        if activations1[x][y] >= -1e-6: 
                            derivatives[x][y] = value1[x][y]
                        else: 
                            derivatives[x][y] = 0.0
                return derivatives
            elif len(activations1.shape) == 3: 
                for k in range(len(activations1)):
                    for x in range(len(activations1[0])):
                        for y in range(len(activations1[0][0])): 
                            if activations1[k][x][y] >= -1e-6: 
                                derivatives[k][x][y] = value1[k][x][y]
                            else: 
                                derivatives[k][x][y] = 0.0
                return derivatives
        elif layerType == "Convolution2D" and (len(activations0.shape)==2) and (len(activations1.shape)==3):
            for x in range(len(activations0)):
                for y in range(len(activations0[0])):
                    if layer >= 0 or (x,y) not in manipulated: 
                        nextValue = 0 
                        for k in range(len(activations1)):
                            for x1 in range(filterSize): 
                                for y1 in range(filterSize): 
                                    if x-2+x1 >=0 and y-2+y1 >= 0 and x-2+x1 < len(activations1[0]) and  y-2+y1 < len(activations1[0][0]): 
                                        d1 = value1[k][x-2+x1][y-2+y1]
                                        d2 = abs(filterCollection[0,k][x1][y1] * d1)
                                        if d2 > nextValue: 
                        	                nextValue = d2
                        derivatives[x][y] = nextValue
            return derivatives
        elif layerType == "Convolution2D" and (len(activations0.shape)==3) and (len(activations1.shape)==3):
            #print activations0.shape, derivatives.shape, activations1.shape, value1.shape
            for l in range(len(activations0)):
                for x in range(len(activations0[0])):
                    for y in range(len(activations0[0][0])):
                        if layer >=0 or (l,x,y) not in manipulated: 
                            nextValue = 0 
                            for k in range(len(activations1)):
                                for x1 in range(filterSize): 
                                    for y1 in range(filterSize): 
                                        if x-2+x1 >=0 and y-2+y1 >= 0 and x-2+x1 < len(activations1[0]) and  y-2+y1 < len(activations1[0][0]): 
                                            d1 = value1[k][x-2+x1][y-2+y1]
                                            d2 = abs(filterCollection[l,k][x1][y1] * d1)
                                            if d2 > nextValue: 
                        	                    nextValue = d2
                            derivatives[l][x][y] = nextValue
            return derivatives
        elif layerType == "Dense" :
            for l in range(len(activations0)):
                nextValue = 0 
                for k in range(len(activations1)):
                    d1 = value1[k]
                    d2 = abs(filterCollection[l,k] * d1)
                    if d2 > nextValue: 
                        nextValue = d2
                derivatives[l] = nextValue
            return derivatives
        else: 
            print "derivative: new layer type: " + str(layerType)
            
            