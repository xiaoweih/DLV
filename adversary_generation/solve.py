#!/usr/bin/env python



import numpy as np
import ast

from z3 import *

import display

set_param(max_lines=1, max_width=1000000)

debug = False

def solve(model,f,box,sgn):


    # number of clauses
    c = 0
    # number of variables 
    d = 0

    (wv,bv) = getWeightVector(model)
    
    
    # get all nodes which has incoming weights
    usefulNodes = list(sorted (set ((zip (*wv))[1])))
    
    # get all nodes
    allNodes = list(sorted (set ((zip (*wv))[0] + (zip (*wv))[1])))
    
    # get all input nodes
    inputNodes = [(l,n) for (l,n) in allNodes if l ==0 ]
    
    # get all output nodes
    outputNodes = list(sorted ((set ((zip (*wv))[1])) - set ((zip (*wv))[0])))
    
    # get all intermediate nodes
    intermediateNodes = list(sorted (set(usefulNodes) - set(outputNodes)))
    
    # create dictionary to index variables
    variables = {}
    i = 0
    for (layerIndex,nodeIndex) in allNodes: 
        variables[layerIndex,nodeIndex,0] = i
        i += 1
        if ((layerIndex,nodeIndex) in usefulNodes): 
            variables[layerIndex,nodeIndex,1] = i
            i += 1
    d = i
    
    # create dictionary to index parameters
    parameters = {}
    j = 0 
    for (layerIndex2,nodeIndex2) in usefulNodes: 
        for ((layerIndex1,nodeIndex1),(l2,n2),p) in getIncomingConnections(wv,(layerIndex2,nodeIndex2)): 
            parameters[layerIndex1,nodeIndex1,layerIndex2,nodeIndex2] = j 
            j += 1
            
    s = Tactic('qflra').solver()
    
    X = [ Real('x%s' % k) for k in range(i)]
    Y = [ Real('y%s' % k) for k in range(j)]
    if debug: f.write("vaiables information: "+"\n")
    if debug: f.write(str(X)+"\n")
    #if debug: print Y
    #if debug: print parameters
    
    
    if debug: f.write("connection nodes: "+"\n")
    for (layerIndex2,nodeIndex2) in usefulNodes: 
        incomingConnections = getIncomingConnections(wv,(layerIndex2,nodeIndex2))
        e = "0"
        for ((l1,n1),(l2,n2),p) in incomingConnections: 
            if l1 > 0 and p != 0 : 
               # e += " + Y[" + str(parameters[l1,n1,layerIndex2,nodeIndex2]) + "] * " + " X["+ str(variables[l1,n1,1]) +"]"
               e += " + " + str(p) + " * " + " X["+ str(variables[l1,n1,1]) +"]"

            elif p != 0 : 
               # e += " + Y[" + str(parameters[l1,n1,layerIndex2,nodeIndex2]) + "] * " + " X["+ str(variables[l1,n1,0]) +"]"
               e += " + " + str(p) + " * " + " X["+ str(variables[l1,n1,0]) +"]"

            #g = "Y[" + str(parameters[l1,n1,layerIndex2,nodeIndex2]) + "] == " + str(p)
            #s.add(eval(g))
            #if debug: print g
        bv1 = [b for (l,n,b) in bv if l==layerIndex2 and n == nodeIndex2]
        if len(bv1) == 0:
            e += " == " + " X["+ str(variables[l2,n2,0]) +"]"
        else: e += "+" + str(bv1[0]) + " == " + " X["+ str(variables[l2,n2,0]) +"]"
        s.add(simplify(eval(e)))
        c += 1
        if debug: f.write(e+"\n")

    for (l,n) in usefulNodes: 
        f1 = "And(X["+ str(variables[l,n,0]) +"] >0, " + "X["+ str(variables[l,n,0]) +"] == X["+ str(variables[l,n,1]) +"])"
        f2 = "And(X["+ str(variables[l,n,0]) +"] <= 0, " + "X["+ str(variables[l,n,1]) +"] == 0)"
        f0 = "Or("+ f1 + "," + f2 + ")"
        if debug: f.write("intermediate nodes: "+"\n")
        if debug: f.write(f0+"\n")
        s.add(eval(f0))
        c += 1 
        
    # conditions for input nodes
    for (l,n) in inputNodes: 
    	if n == 1: f0 = "And(X["+ str(variables[l,n,0]) +"] > "+ str(box[0][0])+", X["+ str(variables[l,n,0]) +"] < "+str(box[1][0])+")" 
    	if n == 2: f0 = "And(X["+ str(variables[l,n,0]) +"] > "+str(box[0][1])+", X["+ str(variables[l,n,0]) +"] < "+str(box[1][1])+")" 
    	s.add(eval(f0))
    	c += 1
        if debug: f.write("input nodes: "+"\n")
        if debug: f.write(f0+"\n")


    # conditions for output nodes
    for (l,n) in outputNodes: 
        bound = []
        if sgn == 2: 
            f0 = "X["+ str(variables[l,2,1]) +"] > X["+ str(variables[l,1,1]) +"]" 
        elif sgn == 1: 
            f0 = "X["+ str(variables[l,1,1]) +"] > X["+ str(variables[l,2,1]) +"]" 

    	#if n==1: f0 = "And(X["+ str(variables[l,n,1]) +"] > "+str(bound[0])+", X["+ str(variables[l,n,1]) +"] < "+str(bound[1])+")" 

    	s.add(eval(f0))
    	c += 1
        if debug: f.write("output nodes: "+"\n")
        if debug: f.write(f0+"\n")

    inputVars = [ eval("X["+str(variables[l,n,0])+"]" ) for (l,n) in inputNodes ]
    if s.check() == sat: 
        cex = [ [x, getDecimalValue(s.model()[x])] for x in inputVars] 
        return [True, cex]
    else: return [False, []]
    #print "statistics for the last check method..."
    #print s.statistics()
    #print(s.model())
    #if debug: f.write(str(s)+"\n")
        
    #print "Number of variables: " + str(d)
    #print "Number of clauses: " + str(c)

def getDecimalValue(v0): 
    v = RealVal(str(v0))
    return float(v.numerator_as_long())/v.denominator_as_long()

def getIncomingConnections(wv,(layerIndex2,nodeIndex2)): 

    return [((l1,n1),(l2,n2),p) for ((l1,n1),(l2,n2),p) in wv if l2==layerIndex2 and n2==nodeIndex2]

def getWeightVector(model):
    weightVector = []
    biasVector = []

    for layer in model.layers:
    	 index=model.layers.index(layer)
         h=layer.get_weights()

         if len(h) > 0: 
             ws = h[0]
             bs = h[1]
             
             # number of nodes in the previous layer
             m = len(ws)
             # number of nodes in the current layer
             n = len(ws[0])
             
             for j in range(1,n+1):
                 biasVector.append((index,j,h[1][j-1]))
             
             for i in range(1,m+1):
                 v = ws[i-1]
                 for j in range(1,n+1): 
                     weightVector.append(((index-1,i),(index,j),v[j-1]))

    return (weightVector,biasVector)        
    
    