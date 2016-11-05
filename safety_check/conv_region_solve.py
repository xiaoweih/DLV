#!/usr/bin/env python

"""
author: Xiaowei Huang
"""

import numpy as np
import math
import ast
import copy
import random
import time
import multiprocessing
import stopit

from z3 import *

#import display
import mnist as mm

from scipy import ndimage
from configuration import *


def conv_region_solve(nfeatures,nfilters,filters,bias,activations0,activations1,cl2,gl2,inds,nn):  

    random.seed(time.time())

    # number of clauses
    c = 0
    # number of variables 
    d = 0

    # variables to be used for z3
    variable={}    
    
    # get the size of the image in the previous layer
    if nfeatures == 1: activations0 = np.expand_dims(activations0, axis=0)
    imageSizeX0 = len(activations0[0])
    imageSizeY0 = len(activations0[0][0])
    
    # get the size of the image in the next layer
    if nfilters == 1: activations1 = np.expand_dims(activations1, axis=0)
    imageSizeX1 = len(activations1[0])
    imageSizeY1 = len(activations1[0][0])
    
    s = Tactic('lra').solver()
    span = {}
    numSpan = {}
    if len((cl2.keys())[0]) == 2: 
        for (x,y) in cl2.keys():
            span[(0,x,y)] = copy.deepcopy(cl2[(x,y)])
            numSpan[(0,x,y)] = copy.deepcopy(gl2[(x,y)])
            #del span[(x,y)]
            #del numSpan[(x,y)]
    else: 
        span = copy.deepcopy(cl2)
        numSpan = copy.deepcopy(gl2)

    if inverseFunction == "area" :
        inds = [(k,x-x1,y-y1) for k in range(nfilters) for (l,x,y) in span.keys() for x1 in range(filterSize) for y1 in range(filterSize) if x-x1 >= 0 and y-y1 >= 0 and x-x1 < activations1.shape[1] and y-y1 < activations1.shape[2] ]

    nextSpan = {}
    nextNumSpan = {}
    for k in inds:
        nextSpan[k] = np.max(span.values())
        nextNumSpan[k] = np.max(numSpan.values())
                   
    # the value of the second layer
    for (k,x,y) in inds: 
        # FIXME: False to skip computation
        while (True): 
            s.reset()
            variable[1,k+1,x,y] = Real('y_%s_%s_%s' % (k+1,x,y))
            str11 = "variable[1,%s,%s,%s] <= %s"%(k+1, x, y, activations1[k][x][y] + nextSpan[(k,x,y)] * nextNumSpan[(k,x,y)])
            str12 = "variable[1,%s,%s,%s] >= %s "%(k+1,x,y, activations1[k][x][y] - nextSpan[(k,x,y)] * nextNumSpan[(k,x,y)])
            str1 = "And(%s,%s)"%(str11,str12)
                    
            forallVarlist = ""
            precond = ""
            existsVarList = "variable[1,%s,%s,%s]"%(k+1,x,y)
            postcond = "variable[1,%s,%s,%s] ==  "%(k+1,x,y)
            for l in range(nfeatures):
                for x1 in range(filterSize):
                    for y1 in range(filterSize): 
                        if (l,x+x1,y+y1) in span.keys(): 
                            variable[0,l+1,x+x1,y+y1] = Real('x_%s_%s_%s' % (l+1,x+x1,y+y1))
                            str21 = "variable[0,%s,%s,%s] <= %s "%(l+1,x+x1,y+y1, activations0[l][x+x1][y+y1] + span[l,x+x1,y+y1] * numSpan[l,x+x1,y+y1])

                            str22 = "variable[0,%s,%s,%s] >= %s "%(l+1,x+x1,y+y1,activations0[l][x+x1][y+y1] - span[l,x+x1,y+y1] * numSpan[l,x+x1,y+y1])

                            str2 = "And(%s,%s)"%(str21,str22)
                            if precond == "": 
                                precond = str2
                            else: precond = "And(%s,%s)"%(precond,str2) 
                            if forallVarlist == "":
                                forallVarlist = "[variable[0,%s,%s,%s]"%(l+1,x+x1,y+y1)
                            else: 
                                forallVarlist += ",variable[0,%s,%s,%s]"%(l+1,x+x1,y+y1)

                            str32 = " variable[0,%s,%s,%s] "%(l+1,x+x1,y+y1)
                            str32 += "* %s + "%(filters[l,k][x1][y1])
                        elif x+x1 < activations0.shape[1] and y+y1 < activations0.shape[2]: 
                            str32 = " %s + "%(activations0[l][x+x1][y+y1] * filters[l,k][x1][y1])
                        postcond += str32
            postcond += str(bias[l,k])
                    
            forallVarlist += "]"
            formula = "ForAll(%s, Implies(%s, Exists(%s, And(%s,%s))))"%(forallVarlist,precond,existsVarList,str1,postcond)

            s.add(eval(formula))

            # FIXME: want to impose timeout on a sinextNumSpane z3 run, 
            # but does not take effect ....

            p = multiprocessing.Process(target=s.check)
            p.start()

            # Wait for timeout seconds or until process finishes
            timeout = 120
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
                    #print "found a region with value numSpan = " + str(nextNumSpan[k,x,y])
                    break
                else:
                    #print "unsatisfiable!" + str(nextNumSpan[(k,x,y)])
                    nextNumSpan[(k,x,y)] = nextNumSpan[(k,x,y)] + 1
            else:
                print "timeout!"
                break
        
    nextNumSpan2 = {}
    nextSpan2 = {}
    for i in range(nn): 
        ind = max(nextNumSpan, key=nextNumSpan.get)
        nextNumSpan2[ind] = nextNumSpan[ind]
        del nextNumSpan[ind]
        nextSpan2[ind] = nextSpan[ind]
    
    #print "found a region: " + str(nextNumSpan2)
    #print(str(nextNumSpan))
    return (nextSpan2,nextNumSpan2)


