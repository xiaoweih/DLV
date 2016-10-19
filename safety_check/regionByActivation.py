#!/usr/bin/env python

"""
compupute e_k according to e_{k-1} and p_{k-1}
author: Xiaowei Huang
"""

import numpy as np
import copy
from scipy import ndimage

from configuration import * 
from imageNet_network import addZeroPadding2D
from networkBasics import * 

    
############################################################
#
#  initialise a region for the input 
#
################################################################   

 
def initialiseRegionActivation(model,manipulated,image): 

    config = NN.getConfig(model)

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
            ls = getTop2DActivation(image,manipulated,[],numDimsToMani,-1)

        elif len(image.shape) == 3:
            # decide how many elements in the input will be considered
            if len(image)*len(image[0])*len(image[0][0])  < featureDims : 
                numDimsToMani = len(image)*len(image[0])*len(image[0][0])
            else: numDimsToMani = featureDims
            # get those elements with maximal/minimum values
            ls = getTop3DActivation(image,manipulated,[],numDimsToMani,-1)     
                
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
        ls = getTopActivation(image,manipulated,-1,numDimsToMani)
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
            ls = getTop2DActivation(image1,manipulated,[],numDimsToMani,-1)

        elif len(image1.shape) == 3:
            # decide how many elements in the input will be considered
            if len(image1)*len(image1[0])*len(image1[0][0])  < featureDims : 
                numDimsToMani = len(image1)*len(image1[0])*len(image1[0][0])
            else: numDimsToMani = featureDims
            # get those elements with maximal/minimum values
            ls = getTop3DActivation(image1,manipulated,[],numDimsToMani,-1)         
        for i in ls: 
            nextSpan[i] = span
            nextNumSpan[i] = numSpan
        
    else: 
        print "initialiseRegionActivation: Unknown layer type ... "
        
    return (nextSpan,nextNumSpan,numDimsToMani)
    
    

    
############################################################
#
#  auxiliary functions
#
################################################################
    
# This function only suitable for the input as a list, not a multi-dimensional array

def getTopActivation(image,manipulated,layerToConsider,numDimsToMani): 

    avoid = repeatedManipulation == "disallowed"
    
    avg = np.sum(image)/float(len(image))
    nimage = list(map(lambda x: abs(avg - x),image))

    topImage = {}
    toBeDeleted = []
    for i in range(len(image)):
        if len(topImage) < numDimsToMani and ((not avoid) or (i not in manipulated)): 
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
    
def getTop2DActivation(image,manipulated,ps,numDimsToMani,layerToConsider): 

    avoid = repeatedManipulation == "disallowed"

    avg = np.sum(image)/float(len(image)*len(image[0]))
    nimage = copy.deepcopy(image)
    for i in range(len(image)): 
        for j in range(len(image[0])):
            nimage[i][j] = abs(avg - image[i][j])
    topImage = {}
    toBeDeleted = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            if len(topImage) < numDimsToMani: 
                topImage[(i,j)] = nimage[i][j]
            else: 
                bl = False 
                for (k1,k2), v in topImage.iteritems():
                    if v < nimage[i][j] and not ((k1,k2) in toBeDeleted) and ((not avoid) or ((i,j) not in manipulated)):  
                        toBeDeleted.append((k1,k2))
                        bl = True
                        break
                if bl == True: 
                    topImage[(i,j)] = nimage[i][j]
    for (k1,k2) in toBeDeleted: 
        del topImage[(k1,k2)]
    return topImage.keys()

# ps are indices of the previous layer

def getTop3DActivation(image,manipulated,ps,numDimsToMani,layerToConsider): 

    avoid = repeatedManipulation == "disallowed"

    avg = np.sum(image)/float(len(image)*len(image[0]*len(image[0][0])))
    nimage = copy.deepcopy(image)
    for i in range(len(image)): 
        for j in range(len(image[0])):
            for k in range(len(image[0][0])):
                nimage[i][j][k] = abs(avg - image[i][j][k])
                
    # do not care about the first dimension
    # only care about individual convolutional node
    if len(ps) > 0: 
        if len(ps[0]) == 3: 
            (p1,p2,p3) = zip(*ps)
            ps = zip(p2,p3)
    ks = []
    pointsToConsider = []
    for i in range(numDimsToMani): 
        if i <= len(ps) - 1: 
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
                    topImage[(i,j,k)] = nimage[i][j][k]
                elif ((j,k) in ps or len(ps) == 0) and (i,j,k) not in ks: 
                    bl = False 
                    for (k1,k2,k3), v in topImage.iteritems():
                        if v < nimage[i][j][k] and not ((k1,k2,k3) in toBeDeleted) and ((not avoid) or ((i,j,k) not in manipulated)):  
                            toBeDeleted.append((k1,k2,k3))
                            bl = True
                            break
                    if bl == True: 
                        topImage[(i,j,k)] = nimage[i][j][k]
    for (k1,k2,k3) in toBeDeleted: 
        del topImage[(k1,k2,k3)]
    return topImage.keys()
