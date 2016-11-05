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


def dense_precision_solve(nfeatures,nfilters,filters,bias,activations0,activations1,span,numSpan,nextSpan,nextNumSpan,pk):  

    # for every dimension k of the next layer, the smallest change 
    # means that it is able to implement the previous layer's manipulations 
    npk = min(nextSpan.values()) 
    for k in nextSpan.keys():
        lst = addexp([0],span.keys(),span,k,filters)
        npk = min(map(abs,lst+[npk]))
    return npk
        
    
    
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
    random.seed(time.time())

    # number of clauses
    c = 0
    # number of variables 
    d = 0

    # variables to be used for z3
    variable={}    
        
    s = Tactic('lra').solver()

    npk = copy.deepcopy(pk)
    factor = 10
    lowered = False
    # the value of the second layer
    for k in nextSpan.keys(): 
        if lowered == True: 
            print("synthesising precision on a point (%s,%s,%s) with precision %s "%(k,x,y,npk))
            lowered = False
                
        # FIXME: false to skip
        while (False): 
            s.reset()
            variable[0,1,k+1] = Real('0_y_%s' % (k+1))
            # v_k \in e_k(A_{i,k})
            str11 = "variable[0,1,"+ str (k+1) + "] <=  " + str(activations1[k] + nextSpan[k] * nextNumSpan[k])
            str12 = "variable[0,1,"+ str (k+1) + "] >=  " + str(activations1[k] - nextSpan[k] * nextNumSpan[k])
            str1 = "And(" + str11 + "," + str12 +")"
            vklist = "variable[0,1,"+ str (k+1) + "]"
                    
            # v_k' \in p_k(v_k)
            variable[1,1,k+1] = Real('1_y_%s' % (k+1))
            str21 = "variable[1,1,"+ str (k+1) + "] <=  variable[0,1,"+ str (k+1) + "] + " + str(npk)
            str22 = "variable[1,1,"+ str (k+1) + "] >=  variable[0,1,"+ str (k+1) + "] - " + str(npk)
            str2 = "And(" + str21 + "," + str22 +")"    
            vkplist = "variable[1,1,"+ str (k+1) + "]"            
                    
            vkmlist = ""
            vkmplist = ""

            str5 = "variable[1,1,"+ str (k+1) + "] ==  "
            for l in span.keys(): 
                # v_{k-1} \in e_{k-1}(A_{i,k-1})
                variable[0,0,l+1] = Real('0_x_%s' % (l+1))
                str31 = "variable[0,0,"+ str (l+1) + "] <=  " + str(activations0[l] + span[l] * numSpan[l])
                str32 = "variable[0,0,"+ str (l+1) + "] >=  " + str(activations0[l] - span[l] * numSpan[l])
                str3 = "And(" + str31 + "," + str32 +")"

                # v_{k-1}' \in p_{k-1}(v_{k-1})
                variable[1,0,l+1] = Real('1_x_%s' % (l+1))
                str41 = "variable[1,0,"+ str (l+1) + "] <=  variable[1,0,"+ str (l+1) + "] + " + str(pk)
                str42 = "variable[1,0,"+ str (l+1) + "] >=  variable[1,0,"+ str (l+1) + "] - " + str(pk)
                str4 = "And(" + str41 + "," + str42 +")"                                
                                
                if vkmlist == "":
                    vkmlist = "[" + "variable[0,0,"+ str (l+1) + "]"
                else: 
                    vkmlist += "," + "variable[0,0,"+ str (l+1) + "]"
                                    
                if vkmplist == "":
                    vkmplist = "[" + "variable[1,0,"+ str (l+1) + "]"
                else: 
                    vkmplist += "," + "variable[1,0,"+ str (l+1) + "]"
                                    
                # v_{k}' = \phi_k(v_{k-1}')
                str51 = " variable[1,0,"+ str(l+1) +"] "
                str51 += "*" + str(filters[l,k]) + " + "
                str5 += str51

            str5 += str(bias[l,k])
                    
            vkmlist += "]"
            vkmplist += "]"

            formula1 = "Exists(" + vkmplist + ", And(" + str4 +"," + str5 +"))"
            formula2 = "ForAll(" + vkplist +", Implies(" + str2+"," + formula1+"))" 
            formula3 = "Exists(" + vklist + ", And(" + str1 +"," + formula2 +"))"
            formula4 = "ForAll(" + vkmlist +", Implies(" + str3+"," + formula3+"))" 

            #s.add(eval("forall("+forallVarlist + "," + formula+")"))
            s.add(eval(formula4))

    
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
                    break
                else:
                    npk = npk / float(factor)
                    lowered = True
                    print "lower the precision to " + str(npk)
            else:
                print "timeout!"
                break
        
    print "the precision is %s. " %(str(npk))
    return npk
"""