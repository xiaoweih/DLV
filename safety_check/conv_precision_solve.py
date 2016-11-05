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


def conv_precision_solve(nfeatures,nfilters,filters,bias,activations0,activations1,span,numSpan,nextSpan,nextNumSpan,pk):  

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

    npk = copy.deepcopy(pk)
    npk = npk
    factor = 10
    lowered = False
    # the value of the second layer
    for k in range(nfilters): 
        for x in range(imageSizeX1):
            for y in range(imageSizeY1): 
                if lowered == True: 
                    print("synthesising precision on a point (%s,%s,%s) with precision %s "%(k,x,y,npk))
                    lowered = False
                
                # FIXME: false to skip
                while (False): 
                    s.reset()
                    variable[0,1,k+1,x,y] = Real('0_y_%s_%s_%s' % (k+1,x,y))
                    # v_k \in e_k(A_{i,k})
                    str11 = "variable[0,1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "] <=  " + str(activations1[k][x][y] + nextSpan * nextNumSpan[k,x,y])
                    str12 = "variable[0,1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "] >=  " + str(activations1[k][x][y] - nextSpan * nextNumSpan[k,x,y])
                    str1 = "And(" + str11 + "," + str12 +")"
                    vklist = "variable[0,1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "]"
                    
                    # v_k' \in p_k(v_k)
                    variable[1,1,k+1,x,y] = Real('1_y_%s_%s_%s' % (k+1,x,y))
                    str21 = "variable[1,1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "] <=  variable[0,1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "] + " + str(npk)
                    str22 = "variable[1,1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "] >=  variable[0,1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "] - " + str(npk)
                    str2 = "And(" + str21 + "," + str22 +")"    
                    vkplist = "variable[1,1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "]"            
                    
                    vkmlist = ""
                    vkmplist = ""

                    str5 = "variable[1,1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "] ==  "
                    for l in range(nfeatures):
                        for x1 in range(3):
                            for y1 in range(3): 
                                # v_{k-1} \in e_{k-1}(A_{i,k-1})
                                variable[0,0,l+1,x+x1,y+y1] = Real('0_x_%s_%s_%s' % (l+1,x+x1,y+y1))
                                str31 = "variable[0,0,"+ str (l+1)  + ","+ str(x+x1) +"," + str(y+y1)+ "] <=  " + str(activations0[l][x+x1][y+y1] + span * numSpan[l,x+x1,y+y1])
                                str32 = "variable[0,0,"+ str (l+1)  + ","+ str(x+x1) +"," + str(y+y1)+ "] >=  " + str(activations0[l][x+x1][y+y1] - span * numSpan[l,x+x1,y+y1])
                                str3 = "And(" + str31 + "," + str32 +")"

                                # v_{k-1}' \in p_{k-1}(v_{k-1})
                                variable[1,0,l+1,x+x1,y+y1] = Real('1_x_%s_%s_%s' % (l+1,x+x1,y+y1))
                                str41 = "variable[1,0,"+ str (l+1)  + ","+ str(x+x1) +"," + str(y+y1)+ "] <=  variable[1,0,"+ str (l+1)  + ","+ str(x+x1) +"," + str(y+y1)+ "] + " + str(pk)
                                str42 = "variable[1,0,"+ str (l+1)  + ","+ str(x+x1) +"," + str(y+y1)+ "] >=  variable[1,0,"+ str (l+1)  + ","+ str(x+x1) +"," + str(y+y1)+ "] - " + str(pk)
                                str4 = "And(" + str41 + "," + str42 +")"                                
                                
                                if vkmlist == "":
                                    vkmlist = "[" + "variable[0,0,"+ str (l+1)  + ","+ str(x+x1) +"," + str(y+y1) + "]"
                                else: 
                                    vkmlist += "," + "variable[0,0,"+ str (l+1)  + ","+ str(x+x1) +"," + str(y+y1) + "]"
                                    
                                if vkmplist == "":
                                    vkmplist = "[" + "variable[1,0,"+ str (l+1)  + ","+ str(x+x1) +"," + str(y+y1) + "]"
                                else: 
                                    vkmplist += "," + "variable[1,0,"+ str (l+1)  + ","+ str(x+x1) +"," + str(y+y1) + "]"
                                    
                                # v_{k}' = \phi_k(v_{k-1}')
                                str51 = " variable[1,0,"+ str(l+1) +"," +str(x) + "+" + str(x1)+"," + str(y)+ "+" + str(y1)+"] "
                                str51 += "*" + str(filters[l,k][x1][y1]) + " + "
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
        
    #print(npk)
    return npk
