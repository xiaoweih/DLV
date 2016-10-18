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


def dense_adversarial_solve(nfeatures,nfilters,filters,bias,input,activations):  

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
    
    #print(filters)
    #print(bias)
        
    s = Tactic('qflra').solver()
    s.reset()

    lower = np.amin(input) 
    upper = np.amax(input) 

    for l in range(nfeatures):
        variable[1,0,l+1] = Real('1_x_%s' % (l+1))
        d += 1
        str2 = "variable[1,0,"+ str(l+1)+"] >= " + str(lower)
        s.add(eval(str2))
        #print(eval(str2))
        c += 1
        str3 = "variable[1,0,"+ str(l+1) +"] <= " + str(upper)
        s.add(eval(str3))
        #print(eval(str3))
        c += 1
                        
    avg = np.divide(np.sum(activations), len(activations))
    corrected = activations - avg                   
    if approach == 1 : (maxdist, topdist) = getTopDis (corrected, num)
    elif approach == -1 : (maxdist, topdist) = getBottomDis (corrected, num)   
                                      
    for k in range(nfilters):
        if k in topdist.keys():         
            variable[1,1,k+1] = Real('1_y_%s' % (k+1))
            d += 1
            string = "variable[1,1,"+ str (k+1)  + "] ==  "
            for l in range(nfeatures):
                newstr1 = " variable[1,0,"+ str(l+1) +"] "
                newstr1 += "*" + str(filters[l,k]) + " + "
                string += newstr1
            string += str(bias[l,k])
            s.add(eval(string))
            #print(eval(string))
            c += 1
                    
            dst1 = eval("variable[1,1,"+ str (k+1)  + "] <= " + str(activations[k]))
            deactivateStr2 = str(activations[k]) +" - "
            deactivateStr2 += "variable[1,1,"+ str (k+1) + "] < " + str(abs(maxdist))+ "/ " + str(changeSpeed)
            dst2 = eval(deactivateStr2)
            dst3 = eval("variable[1,1,"+ str (k+1) + "] < " + str(activations[k]-precision))
            dst = And(dst1, And(dst2, dst3))
                    
            ast1 = eval("variable[1,1,"+ str (k+1)  + "] >= " + str(activations[k]))
            activateStr = "variable[1,1,"+ str (k+1)  + "] - "
            activateStr += str(activations[k]) +" < " + str(abs(maxdist))+ "/ " + str(changeSpeed) 
            ast2 = eval(activateStr)
            ast3 = eval("variable[1,1,"+ str (k+1) + "] > " + str(activations[k]+precision))
            ast = And(ast1, And(ast2, ast3))

            if approach == 1: s.add(dst)
            elif approach == -1: s.add(ast)
            #print(eval(activateStr))
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
            inputVars = [ (l,eval("variable[1,0,"+ str(l+1) + "]")) for l in range(nfeatures)]
            cex = copy.deepcopy(input)
            for (l,x) in inputVars:
                cex[l] = getDecimalValue(s.model()[x])

            print "satisfiable!"
            return (True, cex, activations, [])
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
        if len(dist) < n: 
            dist[i] = image[i]
        else: 
            distance = image[i]
            (k,d) = findMin(dist)
            if distance > d: 
                del dist[k]
                dist[i]=distance
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
        if len(dist) < n: 
            dist[i] = image[i]
        else: 
            distance = image[i]
            (k,d) = findMax(dist)
            if distance < d: 
                del dist[k]
                dist[i]=distance
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
    

     