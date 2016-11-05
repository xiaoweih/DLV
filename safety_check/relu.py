#!/usr/bin/env python

import numpy as np
import math
import ast
import copy
import random
import time
import stopit

from z3 import *

#import display
import mnist as mm

from scipy import ndimage



def bp(input,activations):  

    random.seed(time.time())

    # number of clauses
    c = 0
    # number of variables
    d = 0

    variable={}
        
    (nfeatures,xsize,ysize) = activations.shape

    for l in range(nfeatures): 
        for x in range(xsize):
            for y in range(ysize):
                variable[0,l+1,x,y] = Real('x_%s_%s_%s' % (l+1,x,y))
                d += 1
            
        s = Tactic('qflra').solver()
        s.reset()

        for x in range(xsize):
            for y in range(ysize): 
                variable[1,l+1,x,y] = Real('y_%s_%s_%s' % (l+1,x,y))
                d += 1
                str1 = "And(variable[1,"+ str (l+1)  + ","+ str(x) +"," + str(y)+ "] >= 0, "
                str1 += "variable[1,"+ str (l+1)  + ","+ str(x) +"," + str(y)+ "] ==  "
                str1 += "variable[0,"+ str (l+1)  + ","+ str(x) +"," + str(y)+ "])"

                str2 = "And(variable[1,"+ str (l+1)  + ","+ str(x) +"," + str(y)+ "] < 0, "
                str2 += "variable[0,"+ str (l+1)  + ","+ str(x) +"," + str(y)+ "] < 0) "

                string = " Or(" +str1 + "," + str2 +")"

                s.add(eval(string))
                c += 1

                string = "variable[1,"+ str(l+1) + "," + str(x) +"," + str(y)+ "] ==  " + str(activations[l][x][y]) 
                s.add(eval(string))
                c += 1

        print "Number of variables per feature: " + str(d)
        print "Number of clauses per feature: " + str(c)
    
        if s.check() == sat: 
            inputVars = [ (l,x,y,eval("variable[0,"+ str(l+1) +"," + str(x) +"," + str(y)+ "]")) for x in range(xsize) for y in range(ysize) ]
            cex = {}
            for (l,i,j,x) in inputVars: 
                cex[l,i,j] = getDecimalValue(s.model()[x]) 
                if cex[l,i,j] > 0: 
                    activations[l][i][j] = cex[l,i,j] 
                else:
                    activations[l][i][j] = input[l][i][j] 
        else: 
            print "unsatisfiable! "
            return (False, input)  
                              
    if nfeatures == 1: image = np.squeeze(activations)
    else: image = activations
    return (True, image)

   

def getDecimalValue(v0): 
    v = RealVal(str(v0))
    return long(v.numerator_as_long())/v.denominator_as_long()
    
    