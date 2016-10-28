#!/usr/bin/env python


import numpy as np
import math
import time
import os
import copy


def assure_path_exists(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)

# to check whether a specific point is 
# inconsistent for network and curve
def checkCex(model,x):
    y_predicted = model.predict(np.array([x]))
    y_p = [chooseResult(y) for y in y_predicted]
    y_actual = mapping(x)

    if (y_p[0] == 1) and (y_actual[0] == False): 
        result = True
    elif (y_p[0] == 2) and (y_actual[0] == True): 
        result = True
    else: 
        result = False

    if result == True: 
        print "the point " + str(x) + " is a counterexample!"
    else: 
        print "error: the point " + str(x) + " is NOT a counterexample! please check ... "
    return result
    
def current_milli_time():
    return int(round(time.time() * 1000) % 4294967296)

def diffImage(image1,image2):
    diffnum = 0
    elts = {}
    if len(image1.shape) == 2:
        for x in range(len(image1)):
            for y in range(len(image1[0])):
                if image1[x][y] != image2[x][y]: 
                    diffnum += 1
                    elts[diffnum] = (x,y)
    elif len(image1.shape) == 3:
        for x in range(len(image1)):
            for y in range(len(image1[0])):
               for z in range(len(image1[0][0])):
                  if image1[x][y][z] != image2[x][y][z]: 
                      diffnum += 1
                      elts[diffnum] = (x,y,z)
    elif len(image1.shape) == 1:
        for x in range(len(image1)):
            if image1[x] != image2[x]: 
               diffnum += 1
               elts[diffnum] = x
    return elts
    
    
def euclideanDistance(image1,image2):
    distance = 0
    if len(image1.shape) == 2:
        for x in range(len(image1)):
            for y in range(len(image1[0])):
                if image1[x][y] != image2[x][y]: 
                    distance += (image1[x][y] - image2[x][y]) ** 2
    elif len(image1.shape) == 3:
        for x in range(len(image1)):
            for y in range(len(image1[0])):
               for z in range(len(image1[0][0])):
                  if image1[x][y][z] != image2[x][y][z]: 
                     distance += (image1[x][y][z] - image2[x][y][z]) ** 2

    elif len(image1.shape) == 1:
        for x in range(len(image1)):
            if image1[x] != image2[x]: 
                distance += (image1[x] - image2[x]) ** 2

    return math.sqrt(distance)

def normalisation(y):
    for k in range(len(y)): 
        if y[k] < 0: y[k] = 0 
    return [y[0]/(y[0]+y[1]),y[1]/(y[0]+y[1])]


def chooseResult(y): 
    [y0,y1] = normalisation(y)
    if y0 >= y1: return 1
    else: return 2
    
def addPlotBoxes(plt,boxes,c):
    if len(boxes) > 0: 
        for bb in boxes: 
            addPlotBox(plt,bb,c)
    
def addPlotBox(plt,bb,c): 
        x = [bb[0][0],bb[1][0],bb[1][0],bb[0][0],bb[0][0]]
        y = [bb[0][1],bb[0][1],bb[1][1],bb[1][1],bb[0][1]]
        plt.plot(x,y,c)
        
def equalActivations(activation1,activation2, pk):
    if activation1.shape == activation2.shape :
        if isinstance(activation1, np.float32) or isinstance(activation1, np.float64):
            return abs(activation1 - activation2) < pk
        else: 
            bl = True
            for i in range(len(activation1)):
                bl = bl and equalActivations(activation1[i],activation2[i], pk)
            return bl
    else: print("not the same shape of two activations.")
    
    
############################################################
#
#  auxiliary functions
#
################################################################
    
def getWeight(wv,bv,layerIndex):
    wv = [ (a,(p,c),w) for (a,(p,c),w) in wv if p == layerIndex ]
    bv = [ (p,c,w) for (p,c,w) in bv if p == layerIndex ]
    return (wv,bv)
    
def numberOfFilters(wv):
    return np.amax((zip (*((zip (*wv))[1])))[1])

#  the features of the last layer
def numberOfFeatures(wv):
    return np.amax((zip (*((zip (*wv))[0])))[1])