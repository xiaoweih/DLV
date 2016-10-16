#!/usr/bin/env python


import numpy as np
import time
import os
import copy

from configuration import *
from regionSynth import initialiseRegion

class searchTree:


    # used to store historical images, cls and gls
    # a pair (i,p) is used to represent the index i of the current 
    #  node and its parent node p
    # mfn records the number of features to be manipulated

    def __init__(self, image, k):
        self.images = {}
        self.cls = {}
        self.gls = {}
        self.cps = {}
        self.mfns = {}
        
        # a list of input elements that have been manipulated 
        # we try to avoid update these again
        self.manipulated = {}
        for i in range(-1,k+1):
            self.manipulated[i] = []
        
        # initialisse
        self.images[(-1,-1)] = image
    
        # a queue to be processed first in first out
        self.rk = []
        # the image that is currently processing
        self.crk = (-1,-1)
        
    def destructor(self): 
        self.images = {}
        self.cls = {}
        self.gls = {}
        self.cps = {}
        self.mfns = {}
        self.manipulated = {}
        self.images[(-1,-1)] = []
        self.rk = []        
                
    def getOneUnexplored(self):
        if len(self.rk) > 0: 
            rk0 = self.rk[0]
            self.rk = self.rk[1:]
            self.crk = rk0
            return rk0
        else: return (-1,-1)
    
    def getInfo(self,index):
        return (copy.deepcopy(self.images[index]),self.cls[index],self.gls[index],self.cps[index],self.mfns[index])

    def getHowFar(self,pi,n):
        #if pi >= (numForEachDist ** n): 
        #    return self.getHowFar(pi-(numForEachDist ** n), n+1) 
        #else: return n
        return pi
                
    def parentIndex(self,(ci,pi)): 
        for (k,d) in self.images.keys(): 
            if k == pi: 
                return (k,d)

    def addIntermediateNode(self,image,cl,gl,cp,mfn,index):
        index = (index[0]+1,index[0])
        self.images[index] = image
        self.cls[index] = cl
        self.gls[index] = gl
        self.cps[index] = cp
        self.mfns[index] = mfn
        return index
        
    def addImages(self,model,ims):
        inds = [ i for (i,j) in self.images.keys() if j == -1 ]
        index = max(inds) + 1
        for image in ims: 
            self.images[(index,-1)] = image
            (cl,gl,nn) = initialiseRegion(model,image,self.manipulated[-1])
            # cp : current precision, i.e., p_k
            cp = copy.deepcopy(cp0)
            self.cls[(index,-1)] = cl
            self.gls[(index,-1)] = gl
            self.cps[(index,-1)] = cp
            self.mfns[(index,-1)] = nn
            self.rk.append((index,-1))
            index += 1
            
            manipulated1 = set(self.manipulated[-1])
            manipulated2 = set(self.manipulated[-1] + cl.keys())
            if reset == "onEqualManipulationSet" and manipulated1 == manipulated2: 
                self.clearManipulated(len(self.manipulated))
            else: self.manipulated[-1] = list(manipulated2)
            
            
    def addManipulated(self,k,s):
        self.manipulated[k] = list(set(self.manipulated[k] + s))
            
    def clearManipulated(self,k):
        self.manipulated = {}
        for i in range(-1,k+1):
            self.manipulated[i] = []
                        
    def removeProcessed(self,index):
        children = [ (k,p) for (k,p) in self.cls.keys() if p == index[0] ]
        if children == []: 
            self.removeNode(index)
        else: 
            for childIndex in children:  
                self.removeProcessed(childIndex)
            self.removeNode(index)
                
    def removeNode(self,index):
        del self.images[index]
        del self.cls[index]
        del self.gls[index]
        del self.cps[index]
        del self.mfns[index]
