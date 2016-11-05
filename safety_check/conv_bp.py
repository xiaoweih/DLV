#!/usr/bin/env python

import numpy as np
import math
import ast
import copy
import random
import time
import stopit

from z3 import *

import display
import mnist as mm

from scipy import ndimage



def bp(timeout,nfeatures,nfilters,filters,bias,input,activations,size):  

    random.seed(time.time())

    # number of clauses
    c = 0
    # number of variables 
    d = 0
    
    epsilon = 1.3e-4

    variable={}
    
    if nfeatures == 1: images = np.expand_dims(input, axis=0)
    else: images = input
    (xleft,xright,yleft,yright) = getRegion(images,size)
    
    s = Tactic('qflra').solver()
    s.reset()
    
    for l in range(nfeatures): 
        for x in range(xleft,xright):
            for y in range(yleft,yright):
                variable[0,l+1,x,y] = Real('x_%s_%s_%s' % (l+1,x,y))
                d += 1
                # FIXME: apply only when reach the first layer
                string = "variable[0,"+ str(l+1) + "," + str(x) +"," + str(y)+ "] >= 0 " 
                s.add(eval(string))
                string = "variable[0,"+ str(l+1) + "," + str(x) +"," + str(y)+ "] <= 1  " 
                s.add(eval(string))
                c += 2
    
    xleft2 = copy.copy(xleft)
    xright2 = xright - 2
    yleft2 = copy.copy(yleft)
    yright2 = yright - 2 

    for k in range(nfilters): 
        for x in range(xleft2,xright2):
            for y in range(yleft2,yright2): 
                variable[1,k+1,x,y] = Real('y_%s_%s_%s' % (k+1,x,y))
                d += 1
                string = "variable[1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "] ==  "
                for l in range(nfeatures): 
                    for x1 in range(3):
                        for y1 in range(3): 
                            newstr1 = " variable[0,"+ str(l+1)+","+str(x) + "+" + str(x1)+"," + str(y)+ "+" + str(y1)+"] "
                            newstr1 += "*" + str(filters[l,k][x1][y1]) + " + "
                            string += newstr1
                    string += " + " + str(bias[l,k]) + " + " 
                string += " 0.0 "
                s.add(eval(string))
                c += 1                

                string = "variable[1,"+ str(k+1) + "," + str(x) +"," + str(y)+ "] >  " + str(activations[k][x][y]) + "-" + str(abs(activations[k][x][y])/2)
                s.add(eval(string))
                string = "variable[1,"+ str(k+1) + "," + str(x) +"," + str(y)+ "] <  " + str(activations[k][x][y]) + "+" + str(abs(activations[k][x][y])/2)
                s.add(eval(string))
                c += 2

    print "Number of variables: " + str(d)
    print "Number of clauses: " + str(c)
    

    if s.check() == sat: 
        inputVars = [ (l,x,y,eval("variable[0,"+ str(l+1) +"," + str(x) +"," + str(y)+ "]")) for k in range(nfilters) for l in range(nfeatures) for x in range(xleft2,xright2) for y in range(yleft2,yright2) if (x,y) in dist[k] ]
        cex = {}
        for (l,i,j,x) in inputVars: 
            cex[l,i,j] = getDecimalValue(s.model()[x]) 
            images[l][i][j] = cex[l,i,j] + avg
                                
        if nfeatures == 1: image = np.squeeze(images)
        return (True, image)
    else: 
        print "unsatisfiable! "
        return (False, input)

    

def getDecimalValue(v0): 
    v = RealVal(str(v0))
    return long(v.numerator_as_long())/v.denominator_as_long()
    
def getRegion(images,size):
    image = images[0]
    x = len(image)
    y = len(image[0])
    mx = int(math.floor(x/2))
    my = int(math.floor(y/2))
    dia = int(math.floor(size/2))
    return (mx - dia, mx + dia, my - dia, my + dia)
    