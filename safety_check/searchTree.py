#!/usr/bin/env python

"""
A data structure for organising search

author: Xiaowei Huang
"""

import numpy as np
import time
import os
import copy

from configuration import *
from regionSynth import initialiseRegion

class searchTree:


    # used to store historical images, spans and numSpans
    # a pair (i,p) is used to represent the index i of the current 
    #  node and its parent node p
    # numDimsToMani records the number of features to be manipulated

    def __init__(self, image, k):
        self.images = {}
        self.spans = {}
        self.numSpans = {}
        self.numDimsToManis = {}
        
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
        self.spans = {}
        self.numSpans = {}
        self.numDimsToManis = {}
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
        return (copy.deepcopy(self.images[index]),self.spans[index],self.numSpans[index],self.numDimsToManis[index])

    def getHowFar(self,pi,n):
        #if pi >= (numOfPointsAfterEachFeature ** n): 
        #    return self.getHowFar(pi-(numOfPointsAfterEachFeature ** n), n+1) 
        #else: return n
        return pi
                
    def parentIndex(self,(ci,pi)): 
        for (k,d) in self.images.keys(): 
            if k == pi: 
                return (k,d)

    def addIntermediateNode(self,image,span,numSpan,cp,numDimsToMani,index):
        index = (index[0]+1,index[0])
        self.images[index] = image
        self.spans[index] = span
        self.numSpans[index] = numSpan
        self.numDimsToManis[index] = numDimsToMani
        return index
        
    def addImages(self,model,ims):
        inds = [ i for (i,j) in self.images.keys() if j == -1 ]
        index = max(inds) + 1
        for image in ims: 
            self.images[(index,-1)] = image
            (span,numSpan,nn) = initialiseRegion(model,image,self.manipulated[-1])
            self.spans[(index,-1)] = span
            self.numSpans[(index,-1)] = numSpan
            self.numDimsToManis[(index,-1)] = nn
            self.rk.append((index,-1))
            index += 1
            
            manipulated1 = set(self.manipulated[-1])
            manipulated2 = set(self.manipulated[-1] + span.keys())
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
        children = [ (k,p) for (k,p) in self.spans.keys() if p == index[0] ]
        if children == []: 
            self.removeNode(index)
        else: 
            for childIndex in children:  
                self.removeProcessed(childIndex)
            self.removeNode(index)
                
    def removeNode(self,index):
        del self.images[index]
        del self.spans[index]
        del self.numSpans[index]
        del self.numDimsToManis[index]
