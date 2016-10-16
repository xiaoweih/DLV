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
    cl = {}
    gl = {}
    if len((cl2.keys())[0]) == 2: 
        for (x,y) in cl2.keys():
            cl[(0,x,y)] = copy.deepcopy(cl2[(x,y)])
            gl[(0,x,y)] = copy.deepcopy(gl2[(x,y)])
            #del cl[(x,y)]
            #del gl[(x,y)]
    else: 
        cl = copy.deepcopy(cl2)
        gl = copy.deepcopy(gl2)

    if feedback == "area" :
        inds = [(k,x-x1,y-y1) for k in range(nfilters) for (l,x,y) in cl.keys() for x1 in range(filterSize) for y1 in range(filterSize) if x-x1 >= 0 and y-y1 >= 0 and x-x1 < activations1.shape[1] and y-y1 < activations1.shape[2] ]

    ncl = {}
    ngl = {}
    for k in inds:
        ncl[k] = np.max(cl.values())
        ngl[k] = np.max(gl.values())
                   
    # the value of the second layer
    for (k,x,y) in inds: 
        # FIXME: False to skip computation
        while (True): 
            s.reset()
            variable[1,k+1,x,y] = Real('y_%s_%s_%s' % (k+1,x,y))
            str11 = "variable[1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "] <=  " + str(activations1[k][x][y] + ncl[(k,x,y)] * ngl[(k,x,y)])
            str12 = "variable[1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "] >=  " + str(activations1[k][x][y] - ncl[(k,x,y)] * ngl[(k,x,y)])
            str1 = "And(" + str11 + "," + str12 +")"
                    
            forallVarlist = ""
            precond = ""
            existsVarList = "variable[1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "]"
            postcond = "variable[1,"+ str (k+1)  + ","+ str(x) +"," + str(y)+ "] ==  "
            for l in range(nfeatures):
                for x1 in range(filterSize):
                    for y1 in range(filterSize): 
                        if (l,x+x1,y+y1) in cl.keys(): 
                            variable[0,l+1,x+x1,y+y1] = Real('x_%s_%s_%s' % (l+1,x+x1,y+y1))
                            str21 = "variable[0,"+ str (l+1)  + ","+ str(x+x1) +"," + str(y+y1)+ "] <=  " + str(activations0[l][x+x1][y+y1] + cl[l,x+x1,y+y1] * gl[l,x+x1,y+y1])
                            str22 = "variable[0,"+ str (l+1)  + ","+ str(x+x1) +"," + str(y+y1)+ "] >=  " + str(activations0[l][x+x1][y+y1] - cl[l,x+x1,y+y1] * gl[l,x+x1,y+y1])
                            str2 = "And(" + str21 + "," + str22 +")"
                            if precond == "": 
                                precond = str2
                            else: precond = "And(" + precond +", "+ str2 + ")" 
                            if forallVarlist == "":
                                forallVarlist = "[" + "variable[0,"+ str (l+1)  + ","+ str(x+x1) +"," + str(y+y1) + "]"
                            else: 
                                forallVarlist += "," + "variable[0,"+ str (l+1)  + ","+ str(x+x1) +"," + str(y+y1) + "]"

                            str32 = " variable[0,"+ str(l+1) +"," +str(x) + "+" + str(x1)+"," + str(y)+ "+" + str(y1)+"] "
                            str32 += "*" + str(filters[l,k][x1][y1]) + " + "
                        elif x+x1 < activations0.shape[1] and y+y1 < activations0.shape[2]: 
                            str32 = str(activations0[l][x+x1][y+y1] * filters[l,k][x1][y1]) + " + "
                        postcond += str32
            postcond += str(bias[l,k])
                    
            forallVarlist += "]"
            formula1 = "Implies(" + precond+", Exists("+ existsVarList+ ", And(" + str1 + "," + postcond + ")))"
            formula = "ForAll("+ forallVarlist + ", " + formula1+")"

            #s.add(eval("forall("+forallVarlist + "," + formula+")"))
            s.add(eval(formula))

            # FIXME: want to impose timeout on a single z3 run, 
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
                    #print "found a region with value gl = " + str(ngl[k,x,y])
                    break
                else:
                    #print "unsatisfiable!" + str(ngl[(k,x,y)])
                    ngl[(k,x,y)] = ngl[(k,x,y)] + 1
            else:
                print "timeout!"
                break
        
    ngl2 = {}
    ncl2 = {}
    for i in range(nn): 
        ind = max(ngl, key=ngl.get)
        ngl2[ind] = ngl[ind]
        del ngl[ind]
        ncl2[ind] = ncl[ind]
    
    #print "found a region: " + str(ngl2)
    #print(str(ngl))
    return (ncl2,ngl2)


