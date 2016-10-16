#!/usr/bin/env python

import numpy as np
import math
import ast
import copy
import random
import time
import multiprocessing
import stopit

from z3 import *

import display
import mnist as mm

from scipy import ndimage
from configuration import *


def conv_adversarial_solve(nfeatures,nfilters,filters,bias,input,activations,point):  

    random.seed(time.time())
        
    rn = random.random()
    if rn > float(1)/2: approach = 1 
    else: approach = -1
        
    if approach == 1: 
        print "deactivating the most active ones ... "
    elif approach == -1: 
        print "activating the least active ones ..."
    
    # number of clauses
    c = 0
    # number of variables 
    d = 0

    # variables to be used for z3
    variable={}

    if nfeatures == 1: images = np.expand_dims(input, axis=0)
    else: images = input
    (xleft,xright,yleft,yright) = getRegion(images,point,size)
        
    s = Tactic('qflra').solver()
    s.reset()
    
    xleft2 = copy.copy(xleft)
    xright2 = xright - 2
    yleft2 = copy.copy(yleft)
    yright2 = yright - 2 
                
    lower = {}
    upper = {}
    # the value of the first layer
    for l in range(nfeatures): 
        gap = np.amax(images[l]) - np.amin(images[l])
        upper[l+1] = np.amax(images[l]) 
        lower[l+1] = np.amin(images[l]) 

    for x in range(xleft2,xright2):
        for y in range(yleft2,yright2): 
            for l in range(nfeatures):
                for x1 in range(filterSize):
                    for y1 in range(filterSize): 
                        variable[1,0,l+1,x+x1,y+y1] = Real('1_x_%s_%s_%s' % (l+1,x+x1,y+y1))
                        d += 1
                        str2 = "variable[1,0,"+ str(l+1) +"," +str(x+x1)+"," + str(y+y1)+"] >= " + str(lower[l+1])
                        s.add(eval(str2))
                        c += 1
                        str3 = "variable[1,0,"+ str(l+1) +"," +str(x+x1)+"," + str(y+y1)+"] <= " + str(upper[l+1])
                        s.add(eval(str3))
                        c += 1
                        
    dist = {}
    avg = {}
    for k in range(nfilters):
        avg[k] = np.divide( np.sum(activations[k]), (len(activations[k]) * len(activations[k][0])))
        corrected = activations[k] - avg[k]
        if approach == 1 : (maxdist, topdist) = getTopDis (corrected, num)
        elif approach == -1 : (maxdist, topdist) = getBottomDis (corrected, num)
        dist[k] = topdist
            
        for x in range(xleft2,xright2):
            for y in range(yleft2,yright2): 
                if (x,y) in topdist:         
                    variable[1,1,k+1,x,y] = Real('1_y_%s_%s_%s' % (k+1,x,y))

                    d += 1
                    string = "variable[1,1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "] ==  "
                    for l in range(nfeatures):
                        for x1 in range(filterSize):
                            for y1 in range(filterSize):                                 
                                newstr1 = " variable[1,0,"+ str(l+1) +"," +str(x+x1)+"," + str(y+y1)+"] "
                                newstr1 += "*" + str(filters[l,k][x1][y1]) + " + "
                                string += newstr1
                                d += 1
                    string += str(bias[l,k])
                    s.add(eval(string))
                    c += 1
                    
                    deactivateStr = "And("
                    deactivateStr += "variable[1,1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "] <= "
                    deactivateStr += str(activations[k][x][y]) +" ,  "
                    deactivateStr += str(activations[k][x][y]) +" - "
                    deactivateStr += "variable[1,1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "] < " + str(abs(maxdist))+ "/ " + str(changeSpeed) +" )"
                    
                    activateStr = "And("
                    activateStr += "variable[1,1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "] >= "
                    activateStr += str(activations[k][x][y]) +" ,  "
                    activateStr += "variable[1,1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "] - "
                    activateStr += str(activations[k][x][y]) +" < " + str(abs(maxdist))+ "/ " + str(changeSpeed) +" )"
                    
                    if approach == 1: s.add(eval(deactivateStr))
                    elif approach == -1: s.add(eval(activateStr))
                    c += 1


    print "Number of variables: " + str(d)
    print "Number of clauses: " + str(c)

    p = multiprocessing.Process(target=s.check)
    p.start()

    # Wait for timeout seconds or until process finishes
    p.join(timeout)

    # If thread is still active
    if p.is_alive():
        print "Solver running more than timeout seconds (default="+str(timeout)+"s)! Skip it"
        p.terminate()
        p.join()
    else:
        s_return = s.check()

    if 's_return' in locals():
        if s_return == sat:
            inputVars = [ (l,x,y,x1,y1,eval("variable[1,0,"+ str(l+1) +"," + str(x+x1) +"," + str(y+y1)+ "]")) for k in range(nfilters) for l in range(nfeatures) for x in range(xleft2,xright2) for y in range(yleft2,yright2) for x1 in range(filterSize) for y1 in range(filterSize) if (x,y) in dist[k] ]
            cex = {}
            for (l,i,j,i1,j1,x) in inputVars:
                cex[l,i+i1,j+j1] = getDecimalValue(s.model()[x])
                images[l][i+i1][j+j1] = cex[l,i+i1,j+j1]

            if nfeatures == 1: image = np.squeeze(images)
            else: image = images
            print "satisfiable!"
            return (True, image, activations, [])
        else:
            print "unsatisfiable!"
            return (False, input, activations, [])
    else:
        print "unsatisfiable!"
        return (False, input, activations, [])


def getTopDis(image,n):
    # here we use some randomized strategy 
    # to find the element to modify 
    m = 4*n
    dist = {}
    dist1 = {}
    # remember the minimum distance of the selected points
    for i in range(len(image)):
        for j in range(len(image[0])): 
            if len(dist) < n: 
                distance = image[i][j]
                dist[(i,j)] = distance
            else: 
                distance = image[i][j]
                (k,d) = findMin(dist)
                if distance > d: 
                    del dist[k]
                    dist[(i,j)]=distance
    distlist = random.sample(dist,n)
    for key,item in dist.iteritems(): 
        if (key in distlist): 
            dist1[key] = dist[key]
    return (findMin(dist1)[1],dist1)
    
def getBottomDis(image,n):
    # here we use some randomized strategy 
    # to find the element to modify 
    m = 4*n
    dist = {}
    dist1 = {}
    # remember the minimum distance of the selected points
    for i in range(len(image)):
        for j in range(len(image[0])): 
            if len(dist) < n: 
                distance = image[i][j]
                dist[(i,j)] = distance
            else: 
                distance = image[i][j]
                (k,d) = findMax(dist)
                if distance < d: 
                    del dist[k]
                    dist[(i,j)]=distance
    distlist = random.sample(dist,n)
    for key,item in dist.iteritems(): 
        if (key in distlist): 
            dist1[key] = dist[key]
    return (findMax(dist1)[1],dist1)
                    
def findMin(d):
    k = min(d, key=d.get)
    return (k,d[k])
    
def findMax(d):
    k = max(d, key=d.get)
    return (k,d[k])

def getDecimalValue(v0): 
    v = RealVal(str(v0))
    return long(v.numerator_as_long())/v.denominator_as_long()
    
def getRegion(images,(mx,my),size):
    return (mx - size, mx + size, my - size, my + size)
     