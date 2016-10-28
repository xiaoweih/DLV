#!/usr/bin/env python

"""
author: Xiaowei Huang
"""

import numpy as np
import time
import copy
import math
import random
from operator import mul

import matplotlib.pyplot as plt
from scipy import ndimage

from conv_safety_solve import conv_safety_solve
from conv_adversarial_solve import conv_adversarial_solve
from dense_adversarial_solve import dense_adversarial_solve
from dense_safety_solve import dense_safety_solve
from flatten_safety_solve import flatten_safety_solve
from maxpooling_safety_solve import maxpooling_safety_solve

import conv_bp
import relu

from basics import *
from networkBasics import *
from configuration import * 


def safety_analysis(model,dataset,layer2Consider,imageIndex,st,index,cl2,gl2,cp):

    originalIndex = copy.deepcopy(index)
    (originalImage,prevSpan,prevNumSpan,numDimsToMani) = st.getInfo(index)
    howfar = st.getHowFar(index[0],0)
    
    config = NN.getConfig(model)
        
    # get weights and bias of the entire trained neural network
    (wv,bv) = NN.getWeightVector(model,layer2Consider)
    
    # save the starting layer 
    originalLayer2Consider = copy.deepcopy(layer2Consider)
    
    # predict with neural network
    (originalSpanass,originalConfident) = NN.predictWithImage(model,originalImage)

    classstr = "the right class is " + (str(dataBasics.LABELS(int(originalSpanass))))
    print classstr
    classstr = "the confidence is " + (str(originalConfident))
    print classstr

    if tempFile == "enabled":
        #dataBasics.save(index[0],originalImage,directory_pic_string+"/temp%s_%s.png"%(layer2Consider,index))
        dataBasics.save(index[0],originalImage,directory_pic_string+"/temp.png")
        print "please refer to the file "+directory_pic_string+"/temp.png for the image under processing ..."

    #print "please refer to the file "+directory_pic_string+"/temp%s_%s.png for the image under processing ..."%(layer2Consider,index)
    print "safety analysis for layer %s ... "%(layer2Consider)

    # wk is the set of inputs whose classes are different with the original one
    wk = []
    # wk is for the set of inputs need to be further considered
    rk = [(originalImage,originalConfident)]
    rkupdated = False
    # rs remembers how many points have been tested in this round
    rs = 0
    
    # decide new span,numSpan according to precision cp
    (span,numSpan) = (cl2,gl2) 
    #(span,numSpan) = decideNewP(cl2,gl2,cp)
    #print "the numbers of spans are updated into %s ... "%(numSpan)
    
    originalSpan = copy.deepcopy(span)
    originalNumSpan = copy.deepcopy(numSpan)
    if enumerationMethod == "convex": 
        allRounds = reduce(mul,map(lambda x: 2, numSpan.values()),1)
    elif enumerationMethod == "line": 
        allRounds = reduce(mul,map(lambda x: 2*x + 1, numSpan.values()),1)
    elif enumerationMethod == "point": 
        allRounds = 1
    print "%s regions need to be checked. "%(allRounds)
    
    # counter_numSpan tracks the working point 
    # InitialisedNumSpan remembers 
    (counter_numSpan,InitialisedNumSpan) = initialiseCounter(numSpan)
    counter_numSpans = {}   
    counter_numSpans[originalLayer2Consider] = counter_numSpan 
    round = 0

    cond = False
    rn = 0

    # a recursive procedure until find an interpolated image which is classified 
    # differently with the original image 
    # note: there are some other exit mechanisms in the looping body
    while (not cond) or layer2Consider > 0 :
    
        if layer2Consider == originalLayer2Consider: 
            activations = NN.getActivationValue(model,originalLayer2Consider,originalImage)
            activations1 = imageFromGL(activations,counter_numSpan,span)
            cond = equalCounters(counter_numSpan,numSpan)
            
        nprint("\nin round: %s / %s"%(round, allRounds))
        nprint("layer: " + str(layer2Consider))
        nprint("point %s"%(counter_numSpan))
        #print "maximal point %s"%(numSpan)
        #print "activations1=%s"%(activations1)

        # get the type of the current layer
        layerType = getLayerType(model,layer2Consider)
        #[ lt for (l,lt) in config if layer2Consider == l ]
        #if len(layerType) > 0: layerType = layerType[0]
        #else: print "cannot find the layerType"

        # get the weights and bias for the current layer
        wv2Consider, bv2Consider = getWeight(wv,bv,layer2Consider)
                
        # call different solving approaches according to 
        # the type of the layer and the type of the algorithm
        # FIXME: need to expand this to work with other cases, e.g., MaxPooling2D, Convolution3D
        if layerType == "Convolution2D":
            nprint("convolutional layer, back-propagating ...") 
            if layer2Consider == 0 : 
                activations0 = copy.deepcopy(originalImage)
            else: activations0 = NN.getActivationValue(model,layer2Consider-1,originalImage)
            string = directory_pic_string+"/"+str(imageIndex)+"_original_as_"+str(originalSpanass)
            (bl,newInput) = conv_solve_prep(model,dataBasics,string,originalLayer2Consider,layer2Consider,prevSpan,prevNumSpan,span,numSpan,cp,activations0,wv2Consider,bv2Consider,activations1)
            
        elif layerType == "Dense":  
            nprint("dense layer, back propogation ... ")
            if layer2Consider == 0 : 
                activations0 = copy.deepcopy(originalImage)
            else: activations0 = NN.getActivationValue(model,layer2Consider-1,originalImage)
            string = directory_pic_string+"/"+str(imageIndex)+"_original_as_"+str(originalSpanass)
            (bl,newInput) = dense_solve_prep(model,dataBasics,string,prevSpan,prevNumSpan,span,numSpan,cp,activations0,wv2Consider,bv2Consider,activations1)
            
        elif layerType == "InputLayer":
            nprint("inputLayer layer, back-propagating ... ")
            (bl,newInput) = (True, copy.deepcopy(activations1))
    
        elif layerType == "relu":
            nprint("relu layer, back-propagating ...")
            (bl,newInput) = (True, copy.deepcopy(activations1))

        elif layerType == "ZeroPadding2D":
            nprint("ZeroPadding2D layer, solving ... ")
            image1 = NN.removeZeroPadding2D(activations1)
            (bl,newInput) = (True,image1)

        elif layerType == "MaxPooling2D":
            nprint("MaxPooling2D layer, solving ... ")
            if layer2Consider == 0 : 
                activations0 = copy.deepcopy(originalImage)
            else: activations0 = NN.getActivationValue(model,layer2Consider-1,originalImage)
            image1 = maxpooling_safety_solve(activations0,activations1)
            (bl,newInput) = (True,image1)

        elif layerType == "Flatten":
            nprint("Flatten layer, solving ... ")
            if layer2Consider == 0 : 
                activations0 = copy.deepcopy(originalImage)
            else: activations0 = NN.getActivationValue(model,layer2Consider-1,originalImage)
            image1 = flatten_safety_solve(activations0,activations1)
            (bl,newInput) = (True,image1)
            
        # decide the next step according to the results from the solving
        if bl == False:   
            # if back-propagation fails    
            nprint("back-propagation or solving fails ...")
            layer2Consider = copy.deepcopy(originalLayer2Consider)
            index = copy.deepcopy(originalIndex)
            (image,prevSpan,prevNumSpan,numDimsToMani) = st.getInfo(originalIndex)
            counter_numSpan = counter_numSpans[originalLayer2Consider]
            span = copy.deepcopy(originalSpan)
            numSpan = copy.deepcopy(originalNumSpan)
            (_,InitialisedNumSpan) = initialiseCounter(numSpan)
            counter_numSpan = counterPlusOne(counter_numSpan,numSpan,InitialisedNumSpan)    
            counter_numSpans[originalLayer2Consider] = copy.deepcopy(counter_numSpan)      
            round += 1
                
        elif layer2Consider > 0:       
            # still not yet reach the input layer
            # continue back-propagating 
            layer2Consider -= 1
            activations1 = copy.deepcopy(newInput)
            index = st.parentIndex(index)
            nprint("backtrack to index %s in layer %s"%(index,layer2Consider))
            activations = NN.getActivationValue(model,layer2Consider,originalImage)
            counter_numSpan = getCounter(activations,newInput,prevSpan,prevNumSpan)
            span = copy.deepcopy(prevSpan)
            numSpan = copy.deepcopy(prevNumSpan)
            (image,prevSpan,prevNumSpan,numDimsToMani) = st.getInfo(index)
            counter_numSpans[layer2Consider] = copy.deepcopy(counter_numSpan)
                 
        elif withinRegion(newInput, st) == True:         
            # reached the input layer
            # and has to be within the region
            # check to see if the new input is classified wronextNumSpany.
            rs += 1     
            #print "reach input layer"
            
            if dataset == "imageNet": newInput = normalise(newInput)
            
            nprint("counter: %s"%counter_numSpans[originalLayer2Consider])
            nprint("input: %s"%newInput)


            (newClass,confident) = NN.predictWithImage(model,newInput)            
            if dataset == "twoDcurve": plt.plot([newInput[0]], [newInput[1]], 'g.')

            nprint("confident level: " + str(confident))
            # Great! we found an image which has different class with the original image
            if newClass != originalSpanass: 
                newClassStr = dataBasics.LABELS(int(newClass))
                origClassStr = dataBasics.LABELS(int(originalSpanass))
                classstr = "Class changed! from " + str(origClassStr) +" into " + str(newClassStr)
                print classstr

                path1 = "%s/%s_%s_modified_into_%s_with_confidence_%s.png"%(directory_pic_string,imageIndex,origClassStr,newClassStr,confident)
                dataBasics.save(index[0],newInput, path1)
                
                # add a point whose class is wrong
                wk.append(newInput)
                if exitWhen == "foundFirst": break
                
            else: 
                #oldconf = rk[0][1]
                #if (rk[0][1] == originalConfident): 
                #    rk = [(newInput,confident)]
                #    diffImage(originalImage,newInput)
                #elif confident < oldconf: 
                #    rk = rk + [(newInput,confident)]
                #    diffImage(originalImage,newInput)
                if rkupdated == False: 
                    rk = [(newInput,confident)]
                    if counter_numSpan == numSpan: 
                        rkupdated = True
                        
            layer2Consider = copy.deepcopy(originalLayer2Consider)
            index = copy.deepcopy(originalIndex)
            (image,prevSpan,prevNumSpan,numDimsToMani) = st.getInfo(originalIndex)
            counter_numSpan = counter_numSpans[originalLayer2Consider]
            span = copy.deepcopy(originalSpan)
            numSpan = copy.deepcopy(originalNumSpan)
            (_,InitialisedNumSpan) = initialiseCounter(numSpan)
            counter_numSpan = counterPlusOne(counter_numSpan,numSpan,InitialisedNumSpan)    
            counter_numSpans[originalLayer2Consider] = copy.deepcopy(counter_numSpan)      
            
            round += 1
            rn += 1
            
            #path2 = directory_pic_string+"/temp%s_(round=%s).png"%(rn,round)
            #path2 = directory_pic_string+"/temp.png"
            #dataBasics.save(index[0],newInput, path2)
        else: 
            rs += 1   
            layer2Consider = copy.deepcopy(originalLayer2Consider)
            index = copy.deepcopy(originalIndex)
            (image,prevSpan,prevNumSpan,numDimsToMani) = st.getInfo(originalIndex)
            counter_numSpan = counter_numSpans[originalLayer2Consider]
            span = copy.deepcopy(originalSpan)
            numSpan = copy.deepcopy(originalNumSpan)
            (_,InitialisedNumSpan) = initialiseCounter(numSpan)
            counter_numSpan = counterPlusOne(counter_numSpan,numSpan,InitialisedNumSpan)    
            counter_numSpans[originalLayer2Consider] = copy.deepcopy(counter_numSpan)      
            
            round += 1

    print("ran througn the neural network for %s times."%(rn))
    #path2 = directory_pic_string+"/temp%s_%s.png"%(howfar,originalLayer2Consider)
    #dataBasics.save(newInput, path2)
    
    return (span,numSpan,rs,wk,(zip(*rk))[0])
    
    
############################################################################
#
### decide whether an input is within the region e_0
#
##########################################################################

def withinRegion(newInput,st):
    index = [ (x,y) for (x,y) in st.spans.keys() if y == -1 ]
    (image0,span,numSpan,_) = st.getInfo(index[0])
    
    cls = span.keys()
    wr = True
    for l in cls: 
        if dataset == "imageNet":
            return True
            #(x,y,z) = l
            #if x == 0:
            #    wr = wr and (newInput[l] + 103.939 >= 0)
            #    wr = wr and (newInput[l] + 103.939 <= 255)
            #elif x == 1:
            #    wr = wr and (newInput[l] + 116.779 >= 0)
            #    wr = wr and (newInput[l] + 116.779 <= 255)
            #elif x == 2:
            #    wr = wr and (newInput[l] + 123.68 >= 0)
            #    wr = wr and (newInput[l] + 123.68 <= 255)
        else:
            wr = wr and (newInput[l] >= image0[l] - span[l] * numSpan[l] - epsilon)
            wr = wr and (newInput[l] <= image0[l] + span[l] * numSpan[l] + epsilon)

    return wr
    
def normalise(image): 

    for x in range(len(image)): 
        for y in range(len(image[0])): 
            for z in range(len(image[0][0])): 
                if x == 0:
                    if image[x][y][z] + 103.939 > 255:
                        image[x][y][z] = 255 - 103.939
                    if image[x][y][z] + 103.939 < 0:
                        image[x][y][z] = - 103.939                    
                elif x == 1:
                    if image[x][y][z] + 116.779 > 255:
                        image[x][y][z] = 255 - 116.779
                    if image[x][y][z] + 116.779 < 0:
                        image[x][y][z] = - 116.779  
                elif x == 2:
                    if image[x][y][z] + 123.6 > 255:
                        image[x][y][z] = 255 - 123.6
                    if image[x][y][z] + 123.6 < 0:
                        image[x][y][z] = - 123.6  
    return image
    

############################################################################
#
### preparation for solving convolutional layer
#
##########################################################################

def conv_solve_prep(model,dataBasics,string,originalLayer2Consider,layer2Consider,prevSpan,prevNumSpan,span,numSpan,cp,input,wv,bv,activations):

    # filters can be seen as the output of a convolutional layer
    nfilters = numberOfFilters(wv)
    # features can be seen as the inputs for a convolutional layer
    nfeatures = numberOfFeatures(wv)

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
            
    input2 = copy.deepcopy(input)
    
    if originalLayer2Consider > layer2Consider: 
        (bl1,newInput) = conv_safety_solve(layer2Consider,nfeatures,nfilters,filterCollection,biasCollection,input,activations,prevSpan,prevNumSpan,span,numSpan,cp)
    else: 
        #global enumerationMethod
        #enumerationMethodTemp = copy.deepcopy(enumerationMethod)
        #enumerationMethod = "point"
        (bl1,newInput) = conv_safety_solve(layer2Consider,nfeatures,nfilters,filterCollection,biasCollection,input,activations,prevSpan,prevNumSpan,span,numSpan,cp)
        #enumerationMethod = copy.deepcopy(enumerationMethodTemp)
        
    #dataBasics.save(input2, string+"_"+str(point)+".png")
    #print("bl1="+str(bl1)+"   newInput="+str(len(newInput)))

    nprint("completed a round of processing of the entire image ")
    return (bl1,newInput)
    
#######################################################################################
#
# preparation for solving dense layer
#
#####################################################################################

def dense_solve_prep(model,dataBasics,string,prevSpan,prevNumSpan,span,numSpan,cp,input,wv,bv,activations):

    # filters can be seen as the output of a convolutional layer
    nfilters = numberOfFilters(wv)
    # features can be seen as the inputs for a convolutional layer
    nfeatures = numberOfFeatures(wv)

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
   
    (bl1,newInput) = dense_safety_solve(nfeatures,nfilters,filterCollection,biasCollection,input,activations,prevSpan,prevNumSpan,span,numSpan,cp)
        
    nprint("completed a round of processing ")
    return (bl1,newInput)
    
    

############################################################################
#
###  functions for exploring the space
#
##########################################################################

def counterPlusOne(counter_numSpan,numSpan,InitialisedNumSpan):

    if enumerationMethod == "line": 
        return counterPlusOne0(counter_numSpan,numSpan,InitialisedNumSpan)
    elif enumerationMethod == "convex": 
        return counterPlusOne1(counter_numSpan,numSpan,InitialisedNumSpan)
    elif enumerationMethod == "point": 
        return numSpan

## explore outmost spaces 

def counterPlusOne1(counter_numSpan,numSpan,InitialisedNumSpan): 

    j = -1    
    ncounter_numSpan = copy.deepcopy(counter_numSpan)
    for (i, p) in InitialisedNumSpan.iteritems(): 
        if counter_numSpan[p] < numSpan[p]: 
            j = p 
            break
        else: 
            ncounter_numSpan[p] = - numSpan[p]
    if j == -1 : 
        ncounter_numSpan = copy.deepcopy(numSpan)
    else: 
        ncounter_numSpan[j] = numSpan[p]
    return ncounter_numSpan


## explore all possible points in the space

def counterPlusOne0(counter_numSpan,numSpan,InitialisedNumSpan): 

    j = -1    
    ncounter_numSpan = copy.deepcopy(counter_numSpan)
    for (i, p) in InitialisedNumSpan.iteritems(): 
        if counter_numSpan[p] < numSpan[p]: 
            j = p 
            break
        else: 
            ncounter_numSpan[p] = - numSpan[p]
    if j == -1 : 
        ncounter_numSpan = copy.deepcopy(numSpan)
    else: 
        ncounter_numSpan[j] = counter_numSpan[j] + 1
    return ncounter_numSpan


############################################################################
#
### auxiliary functions
#
##########################################################################
    

def imageSize(image):
    return len(image[0])

def imageFromGL(image,counter_numSpan,span):
    nimage = copy.deepcopy(image)
    for (p, c) in counter_numSpan.iteritems():
        nimage[p] = image[p] + c * span[p]
    return nimage    

def equalCounters(gl1,gl2): 
    bl = True
    for (i,p1) in gl1.iteritems(): 
        if not(p1 == gl2[i]): 
            bl = False
            break
    return bl 
           
def decideNewP(span,numSpan,cp): 
    nextSpan = copy.deepcopy(span)
    nextNumSpan = copy.deepcopy(numSpan)
    for (p, v) in numSpan.iteritems(): 
        l = span[p] * v # * 2
        nextSpan[p] = cp
        nextNumSpan[p] = math.ceil(l / float(cp))
    return (nextSpan,nextNumSpan)
    
def initialiseCounter(numSpan):
    counter_numSpan = {}
    InitialisedNumSpan = {}
    j = 0
    for i in numSpan.keys(): 
        counter_numSpan[i] = - numSpan[i]
        InitialisedNumSpan[j] = i
        j += 1
    return (counter_numSpan,InitialisedNumSpan)
    
    
def getCounter(activations,newInput,span,numSpan): 
    #print("%s\n%s\n%s"%(activations.shape,newInput.shape,span.keys()))
    nextNumSpan = copy.deepcopy(numSpan)
    for l in span.keys():
        if len(activations.shape) == 3:
            nextNumSpan[l] = round((newInput[l[0]][l[1]][l[2]] - activations[l[0]][l[1]][l[2]])/span[l], 0)
        elif len(activations.shape) == 2 :
            nextNumSpan[l] = round((newInput[l[0]][l[1]] - activations[l[0]][l[1]])/span[l], 0)
        elif len(activations.shape) == 1 :
            nextNumSpan[l] = round((newInput[l] - activations[l])/span[l], 0)
    return nextNumSpan
    
    
def diffImage(image1,image2):
    i = 0
    if len(image1.shape) == 1: 
        for x in range(len(image1)):
                if image1[x] != image2[x]: 
                    i += 1
                    nprint("dimension %s is changed from %s to %s"%(x,image1[x],image2[x]))
    elif len(image1.shape) == 2:
        for x in range(len(image1)):
            for y in range(len(image1[0])):
                if image1[x][y] != image2[x][y]: 
                    i += 1
                    nprint("dimension (%s,%s) is changed from %s to %s"%(x,y,image1[x][y],image2[x][y]))
    elif len(image1.shape) == 3:
        for x in range(len(image1)):
            for y in range(len(image1[0])):
                for z in range(len(image1[0])):
                    if image1[x][y][z] != image2[x][y][z]: 
                        i += 1
                        nprint("dimension (%s,%s,%s) is changed from %s to %s"%(x,y,z,image1[x][y][z],image2[x][y][z]))
    print("%s elements have been changed!"%i)
    
############################################################################
#
### for back-propagation, not used at the moment
#
##########################################################################

def conv_bp_prep(model,input,wv,bv,activations):

    nfilters = numberOfFilters(wv)
    nfeatures = numberOfFeatures(wv)
    
    print "number of filters: " + str(nfilters)
    print "number of features in the previous layer: " + str(nfeatures)
    
    (_, sizex, sizey) = activations.shape
    sizex += 2

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
                    
            flipedFilter = np.fliplr(np.flipud(filter))
            biasCollection[l,k] = bias
            filterCollection[l,k] = flipedFilter
            #print filter.shape
    
    (bl,newInput) = conv_bp.bp(nfeatures,nfilters,filterCollection,biasCollection,input,activations)
    
    return (bl,newInput)
