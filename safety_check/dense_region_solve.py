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

def dense_region_solve(nfeatures,nfilters,filters,bias,activations0,activations1,span,numSpan,inds):  

    if regionSynthMethod == "stretch": 
        return stretch(nfeatures,nfilters,filters,bias,activations0,activations1,span,numSpan,inds)
    elif regionSynthMethod == "condense":
        return condense(nfeatures,nfilters,filters,bias,activations0,activations1,span,numSpan,inds)


def condense(nfeatures,nfilters,filters,bias,activations0,activations1,span,numSpan,inds):  

    random.seed(time.time())

    # number of clauses
    c = 0
    # number of variables 
    d = 0

    # variables to be used for z3
    variable={}    
    
    s = Solver()
    
    # initialise nextSpan and nextNumSpan for later refinement
    nextSpan = {}
    nextNumSpan = {}
        
    decidedDims = []
    for l in span.keys(): 
        dims = random.sample([ x for x in inds if x not in decidedDims], refinementRate)
        decidedDims = decidedDims + dims
        for k in dims: 
            # the idea of this is as follows:
            #  there are 3^n possible manipulations, and 
            #  for the current k, we select the longest change as its manipulation
            #  so that the region of the next layer can cover the previous layer 
            nextSpan[k] = 0
            for l1 in span.keys():
                nextSpan[k] += abs(span[l1] * numSpan[l1] * filters[l1,k])
            nextSpantemp = 0
            for l1 in range(nfeatures): 
                #print nfeatures, l1
                if l1 not in span.keys():
                    nextSpantemp +=  filters[l1,k] * activations0[l1]
            nextSpantemp += bias[l,k]
            nextSpan[k] += abs(nextSpantemp)
            nextNumSpan[k] = numSpan[l]
            # adjust 
            lst = addexp([0],span.keys(),span,k,filters)
            lst2 = map(abs,lst+[nextSpan[k]])
            npk = findMin(min(lst2),lst2)
            t = math.ceil(nextSpan[k] / npk)
            nextSpan[k] = npk
            nextNumSpan[k] = numSpan[l] * t
    return (nextSpan,nextNumSpan)
    
def findMin(v,lst):

    bl = True
    for l in lst: 
        if l%v > epsilon: bl = False
    if bl == True: 
        return v
    else: 
        return findMin(v / float(2),lst)
    
    
def addexp(lst,cls,span,k,filters):
    # this is to find the smallest change of k
    # with respect to the manipulations of the previous layer
    lst2 = []
    l = cls[0]
    # print lst
    for e in lst: 
        e1 = e + filters[l,k] * span[l]
        e2 = e - filters[l,k] * span[l]
        lst2 = lst2 + [e1,e2]
    remain = cls[1:]
    if len(remain) > 0: 
        return addexp(lst2,remain,span,k,filters)
    else: return lst2

"""
        # a manipulation in the current layer
        variable[0,0,l+1] = Real('0_x_%s' % (l+1))
        variable[1,0,l+1] = Real('1_x_%s' % (l+1))
        varlist00 = "variable[0,0,%s]"%(l+1)
        varlist10 = "[variable[1,0,%s]"%(l+1)
        str001 = "variable[0,0,%s] <= %s "%(l+1, activations0[l] + span[l] * numSpan[l])
        str002 = "variable[0,0,%s] >= %s "%(l+1, activations0[l] - span[l] * numSpan[l])
        str00 = "And(" + str001 + "," + str002 +")"

        str10 = "variable[1,0,%s] ==  variable[0,0,%s] + %s "%(l+1,l+1,span[l])

        # the relation between two layers on the two points: current point and manipulated point
        for k in dims: 
            s.reset()

            variable[0,1,k+1] = Real('0_y_%s' % (k+1))
            variable[1,1,k+1] = Real('1_y_%s' % (k+1))
            varlist11 = ", variable[0,1,%s], variable[1,1,%s]"%(k+1,k+1)
            
            str01 = "True"
            for p in [0,1]: 
                str1 = "variable[%s,1,%s] == "%(p,k+1)
                for l1 in range(nfeatures):
                    if l1 == l:
                        str32 = " variable[%s,0,%s] "%(p,l+1)
                        str32 += "*" + str(filters[l,k]) + " + "
                        str1 += str32
                    else: 
                        str32 = str(activations0[l1]*filters[l1,k])  + " + "
                        str1 += str32
                str1 += str(bias[l,k])
                str01 = "And(%s,%s)"%(str01,str1)
                
            # a manipulation in the next layer
            variable[2,1,k+1] = Int('g_y_%s' % (k+1))
            variable[3,1,k+1] = Real('c_y_%s' % (k+1))
            if filters[l,k] >= 0: 
                str21 = "variable[1,1,%s] <= variable[0,1,%s] + variable[2,1,%s]*variable[3,1,%s] + %s"%(k+1,k+1,k+1,k+1,epsilon)
                str22 = "variable[1,1,%s] >= variable[0,1,%s] + variable[2,1,%s]*variable[3,1,%s] - %s"%(k+1,k+1,k+1,k+1,epsilon)
            else: 
                str21 = "variable[0,1,%s] <= variable[1,1,%s] + variable[2,1,%s]*variable[3,1,%s] + %s"%(k+1,k+1,k+1,k+1,epsilon)
                str22 = "variable[0,1,%s] >= variable[1,1,%s] + variable[2,1,%s]*variable[3,1,%s] - %s"%(k+1,k+1,k+1,k+1,epsilon)
                
            str11 = "And(%s,%s)"%(str21,str22)
            str1 = "And(%s,%s)"%(str01,str11)
            str1 = "And(%s,%s)"%(str1,str10)

            varlist11 += "]"
            varlist1 = varlist10 + varlist11
            formula = "ForAll(%s,Implies(%s, Exists(%s, %s)))"%(varlist00,str00,varlist1,str1)
            s.add(eval(formula))
            print eval(formula)

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
                    inputVars = [ (k,eval("variable[2,1,%s]"%(k+1)),eval("variable[3,1,%s]"%(k+1))) ]
                    for (k,g,c) in inputVars:
                        nextNumSpan[k] = getDecimalValue(s.model()[g])
                        nextSpan[k] = getDecimalValue(s.model()[c])
                else:
                    print "unsatisfiable! Need a fix ... "
            else:
                print "timeout!"
                break
        
    return (nextSpan,nextNumSpan)
"""


def stretch(nfeatures,nfilters,filters,bias,activations0,activations1,span,numSpan,inds):  


    random.seed(time.time())

    # number of clauses
    c = 0
    # number of variables 
    d = 0

    # variables to be used for z3
    variable={}    
    
    s = Tactic('lra').solver()
    
    # initialise nextSpan and nextNumSpan for later refinement
    nextSpan = {}
    nextNumSpan = {}
    for k in inds:
        nextSpan[k] = np.max(span.values())
        nextNumSpan[k] = np.max(numSpan.values())
                
    # the value of the second layer
    for k in inds: 
        while (True): 
            s.reset()
            variable[1,k+1] = Real('y_%s' % (k+1))
            str11 = "variable[1,"+ str (k+1) + "] <=  " + str(activations1[k] + nextSpan[k] * nextNumSpan[k])
            str12 = "variable[1,"+ str (k+1) + "] >=  " + str(activations1[k] - nextSpan[k] * nextNumSpan[k])
            str1 = "And(" + str11 + "," + str12 +")"
                    
            forallVarlist = ""
            precond = ""
            existsVarList = "variable[1,"+ str (k+1) + "]"
            postcond = "variable[1,"+ str (k+1) + "] ==  "
            for l in range(nfeatures):
                if l in span.keys():
                    variable[0,l+1] = Real('x_%s' % (l+1))
                    str21 = "variable[0,"+ str (l+1) + "] <=  " + str(activations0[l] + span[l] * numSpan[l])
                    str22 = "variable[0,"+ str (l+1) + "] >=  " + str(activations0[l] - span[l] * numSpan[l])
                    str2 = "And(" + str21 + "," + str22 +")"
                    if precond == "": 
                            precond = str2
                    else: precond = "And(" + precond +", "+ str2 + ")" 
                    if forallVarlist == "":
                        forallVarlist = "[" + "variable[0,"+ str (l+1) + "]"
                    else: 
                        forallVarlist += "," + "variable[0,"+ str (l+1) + "]"

                    str32 = " variable[0,"+ str(l+1) +"] "
                    str32 += "*" + str(filters[l,k]) + " + "
                    postcond += str32
                else: 
                    str32 = str(activations0[l]*filters[l,k])  + " + "
                    postcond += str32
                    
            postcond += str(bias[l,k])
                    
            forallVarlist += "]"
            formula1 = "Implies(" + precond+", Exists("+ existsVarList+ ", And(" + str1 + "," + postcond + ")))"
            formula = "ForAll("+ forallVarlist + ", " + formula1+")"

            #s.add(eval("forall("+forallVarlist + "," + formula+")"))
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
                    print "found a region for dimension %s with value (span,numSpan) = (%s, %s)"%(k,str(nextSpan[k]),str(nextNumSpan[k]))
                    break
                else:
                    #print "unsatisfiable!" + str(nextNumSpan[(k,x,y)])
                    nextNumSpan[k] = nextNumSpan[k] + 1
            else:
                print "timeout!"
                break
        
    return (nextSpan,nextNumSpan)


