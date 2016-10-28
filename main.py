#!/usr/bin/env python

"""
main file

author: Xiaowei Huang
"""

import sys
sys.path.append('networks')
sys.path.append('safety_check')
sys.path.append('configuration')

import time
import numpy as np
import copy 
import random
import matplotlib.pyplot as plt

from loadData import loadData 
from regionSynth import regionSynth, initialiseRegion
from precisionSynth import precisionSynth
from safety_analysis import safety_analysis

from configuration import *
from basics import *
from networkBasics import *

from searchTree import searchTree
from dataCollection import dataCollection
        
def main():

    model = loadData()
    dc = dataCollection()
    
    # handle a set of inputs starting from an index
    if dataProcessing == "batch": 
        for whichIndex in range(startIndexOfImage,startIndexOfImage + dataProcessingBatchNum):
            print "\n\nprocessing input of index %s in the dataset: " %(str(whichIndex))
            if task == "safety_check": 
                handleOne(model,dc,whichIndex)
    # handle a sinextNumSpane input
    else: 
        print "\n\nprocessing input of index %s in the dataset: " %(str(startIndexOfImage))
        if task == "safety_check": 
            handleOne(model,dc,startIndexOfImage)
    if dataProcessing == "batch": 
        dc.provideDetails()
        dc.summarise()
    dc.close()
      
###########################################################################
#
# safety checking
# starting from the first hidden layer
#
############################################################################

def handleOne(model,dc,startIndexOfImage):

    # get an image to interpolate
    global np
    image = NN.getImage(model,startIndexOfImage)
    print("the shape of the input is "+ str(image.shape))
    #image = np.array([3.58747339,1.11101673])
    
    dc.initialiseIndex(startIndexOfImage)

    if checkingMode == "stepwise":
        k = startLayer
    elif checkingMode == "specificLayer":
        k = maxLayer
        
    while k <= maxLayer: 
    
        layerType = getLayerType(model, k)
        start_time = time.time()
        
        # only these layers need to be checked
        if layerType in ["Convolution2D", "Dense"]: 
    
            dc.initialiseLayer(k)
    
            # initialise a search tree
            st = searchTree(image,k)
            st.addImages(model,[image])
            print "\nstart checking the safety of layer "+str(k)
            print "the current context is %s"%(st.numSpans)
        
            (originalClass,originalConfident) = NN.predictWithImage(model,image)
            origClassStr = dataBasics.LABELS(int(originalClass))   
     
            path0="%s/%s_original_as_%s_with_confidence_%s.png"%(directory_pic_string,startIndexOfImage,origClassStr,originalConfident)
            dataBasics.save(-1,image, path0)

            # for every layer
            f = 0 
            if numOfPointsAfterEachFeature == 1: 
                testNum = numOfFeatures
            else: testNum = (numOfPointsAfterEachFeature ** (n+1)) / (numOfPointsAfterEachFeature - 1)
            while f <= testNum: 

                f += 1
                index = st.getOneUnexplored()
                imageIndex = copy.deepcopy(index)
            
                howfar = st.getHowFar(index[0],0)
                print "\nhow far is the current image from the original one: %s"%(howfar)
            
                # for every image
                # start from the first hidden layer
                t = 0
                while True: 

                    print "\ncurrent index: %s."%(str(index))
                    print "current layer: %s."%(t)

                    print "\nhow many dimensions have been changed: %s."%(len(st.manipulated[-1]))

                    # pick the first element of the queue
                    print "1) get a manipulated input ..."
                    (image0,span,numSpan,numDimsToMani) = st.getInfo(index)
                    
                    path2 = directory_pic_string+"/temp.png"
                    print " saved into %s"%(path2)
                    dataBasics.save(index[0],image0,path2)

                    print "2) synthesise region ..."
                     # ne: next region, i.e., e_{k+1}
                    (nextSpan,nextNumSpan,numDimsToMani) = regionSynth(model,dataset,image0,st.manipulated[t],t,span,numSpan,numDimsToMani)
                    st.addManipulated(t,nextSpan.keys())

                    #print "3) synthesise precision ..."
                    #if not found == True: nextNumSpan = dict(map(lambda (k,v): (k, abs(v-1)), nextNumSpan.iteritems()))
                    # np : next precision, i.e., p_{k+1}
                    #np = precisionSynth(model,dataset,image0,t,span,numSpan,nextSpan,nextNumSpan,cp)
                    (nextSpan,nextNumSpan,np) = precisionSynth(t,nextSpan,nextNumSpan)
                    #print "the precision is %s."%(np)
                    
                    print "dimensions to be considered: %s"%(nextSpan)
                    #print "dimensions that have been considered before: %s"%(st.manipulated[t])
                    print "spans for the dimensions: %s"%(nextNumSpan)
                
                    if t == k: 
                        print "3) safety analysis ..."
                        # wk for the set of counterexamples
                        # rk for the set of images that need to be considered in the next precision
                        # rs remembers how many input images have been processed in the last round
                        # nextSpan and nextNumSpan are revised by considering the precision np
                        (nextSpan,nextNumSpan,rs,wk,rk) = safety_analysis(model,dataset,t,startIndexOfImage,st,index,nextSpan,nextNumSpan,np)

                        print "4) add new images ..."
                        random.seed(time.time())
                        if len(rk) > numOfPointsAfterEachFeature: 
                            rk = random.sample(rk, numOfPointsAfterEachFeature)
                        diffs = diffImage(image0,rk[0])
                        print("the dimensions of the images that are changed in the previous round: %s"%diffs)
                        if len(diffs) == 0: st.clearManipulated(k)
                        st.addImages(model,rk)
                        st.removeProcessed(imageIndex)
                        (re,percent,eudist) = reportInfo(image,rs,wk,numDimsToMani,howfar,image0)
                        break
                    else: 
                        print "3) add new intermediate node ..."
                        index = st.addIntermediateNode(image0,nextSpan,nextNumSpan,np,numDimsToMani,index)
                        re = False
                        t += 1
                if re == True: 
                    dc.addManipulationPercentage(percent)
                    dc.addEuclideanDistance(eudist)
                    (ocl,ocf) = NN.predictWithImage(model,rk[0])
                    dc.addConfidence(ocf)
                    break
                
            st.destructor()
            
        runningTime = time.time() - start_time   
        dc.addRunningTime(runningTime)
        if re == True and exitWhen == "foundFirst": 
            break
        k += 1    
     
    print("Please refer to the file %s for statistics."%(dc.fileName))
    

def reportInfo(image,rs,wk,numDimsToMani,howfar,image0):

    # exit only when we find an adversarial example
    if wk == []:    
        print "no adversarial example is found in this layer."  
        return (False,0,0)
    else: 
        print "an adversarial example has been found."
        diffs = diffImage(image,image0)
        eudist = euclideanDistance(image,image0)
        elts = len(diffs.keys())
        if len(image0.shape) == 2: 
            percent = elts / float(len(image0)*len(image0[0]))
        elif len(image0.shape) == 1:
            percent = elts / float(len(image0))
        elif len(image0.shape) == 3:
            percent = elts / float(len(image0)*len(image0[0])*len(image0[0][0]))
        return (True,percent,eudist)
        
if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    