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


def dense_safety_solve(nfeatures,nfilters,filters,bias,input,activations,pcl,pgl,span,numSpan,pk):  

    random.seed(time.time())
    rn = random.random()
    
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
    for l in pcl.keys():
        variable[1,0,l+1] = Real('1_x_%s' % (l+1))
        d += 1

                        
    for k in span.keys():
        variable[1,1,k+1] = Real('1_y_%s' % (k+1))
        d += 1
        string = "variable[1,1,%s] ==  "%(k+1)
        for l in range(nfeatures):
            if l in pcl.keys(): 
                newstr1 = " variable[1,0,%s] * %s + "%(l+1,filters[l,k])
            else: 
                newstr1 = " %s + "%(input[l]*filters[l,k])
            string += newstr1
        string += str(bias[l,k])
        s.add(eval(string))
        #print(eval(string))
        c += 1
                            
        pStr1 = "variable[1,1,%s] == %s"%(k+1, activations[k])

        s.add(eval(pStr1))
        c += 1

    nprint("Number of variables: " + str(d))
    nprint( "Number of clauses: " + str(c))

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
            inputVars = [ (l,eval("variable[1,0,%s]"%(l+1))) for l in pcl.keys() ]
            cex = copy.deepcopy(input)
            for (l,x) in inputVars:
                cex[l] = getDecimalValue(s.model()[x])
                

            nprint( "satisfiable!")
            return (True, cex)
        else:
            nprint( "unsatisfiable!")
            return (False, input)
    else:
        print "unsatisfiable!"
        return (False, input)


def getDecimalValue(v0): 
    v = RealVal(str(v0))
    return float(v.numerator_as_long())/v.denominator_as_long()
    

     