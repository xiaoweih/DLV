#!/usr/bin/env python

"""
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
from adversarial_analysis import adversarial_analysis
from safety_analysis import safety_analysis

from configuration import *
from basics import *
from networkBasics import *

from searchTree import searchTree
from dataCollection import dataCollection
        
def main():

    (model,dataset,maxilayer,startIndex) = loadData()
    
    dc = dataCollection()
    
    statisticsFile = open('%s/statistics.txt'%(directory_statistics_string), 'a')

    # handle a set of inputs starting from an index
    if dataProcessing == "batch": 
        for whichIndex in range(startIndex,startIndex + dataProcessingBatchNum):
            print "\n\nprocessing input of index %s in the dataset: " %(str(whichIndex))
            if task == "safety_check": 
                # to compare the performance of area and point approaches
                #global feedback
                #feedbackTemp = copy.deepcopy(feedback)
                handleOne(model,dataset,dc,statisticsFile,maxilayer,whichIndex)
                #feedback = "area"
                #handleOne(model,dataset,statisticsFile,maxilayer,whichIndex)
                #feedback = copy.deepcopy(feedbackTemp)
            elif task == "adversary_generation":  
                adversarial_analysis(model,dataset,maxilayer,startIndex,cl0,gl0,cp0)
    # handle a single input
    else: 
        print "\n\nprocessing input of index %s in the dataset: " %(str(startIndex))
        if task == "safety_check": 
            handleOne(model,dataset,dc,statisticsFile,maxilayer,startIndex)
        elif task == "adversary_generation":   
            adversarial_analysis(model,dataset,maxilayer,startIndex,cl0,gl0,cp0)

    statisticsFile.close()
    dc.summarise()
    dc.close()
      
###########################################################################
#
# safety checking
# starting from the first hidden layer
#
############################################################################

def handleOne(model,dataset,dc,statisticsFile,maxilayer,startIndex):

    # ce: the region definition for layer 0, i.e., e_0
    # get an image to interpolate
    global np
    image = NN.getImage(model,startIndex)
    print("the shape of the input is "+ str(image.shape))
    #image = np.array([3.58747339,1.11101673])
    statisticsFile.write("\nimageIndex = %s\n"%(startIndex))
    statisticsFile.write("feedback = %s\n"%(feedback))
    
    dc.initialiseIndex(startIndex)

    if checkingMode == "stepwise":
        k = 0
    elif checkingMode == "specificLayer":
        k = maxilayer
        
    while k <= maxilayer: 
    
        layerType = getLayerType(model, k)
        start_time = time.time()
        
        # only these layers need to be checked
        if layerType in ["Convolution2D", "Dense"]: 
    
            dc.initialiseLayer(k)
    
            # initialise a search tree
            st = searchTree(image,k)
            st.addImages(model,[image])
            print "\nstart checking the safety of layer "+str(k)
            print "the current context is %s"%(st.gls)
        
            (originalClass,originalConfident) = NN.predictWithImage(model,image)
            origClassStr = dataBasics.LABELS(int(originalClass))   
     
            path0="%s/%s_original_as_%s_with_confidence_%s.png"%(directory_pic_string,startIndex,origClassStr,originalConfident)
            dataBasics.save(-1,image, path0)

            # for every layer
            f = 0 
            if numForEachDist == 1: 
                testNum = numOfDist
            else: testNum = (numForEachDist ** (n+1)) / (numForEachDist - 1)
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
                    (image0,cl,gl,cp,mfn) = st.getInfo(index)
                    
                    path2 = directory_pic_string+"/temp%s.png"%(str(index))
                    dataBasics.save(index[0],image0, path2)

                    print "2) synthesise region ..."
                     # ne: next region, i.e., e_{k+1}
                    (ncl,ngl,mfn) = regionSynth(model,dataset,image0,st.manipulated[t],t,cl,gl,cp,mfn)
                    st.addManipulated(t,ncl.keys())


                    #print "3) synthesise precision ..."
                    #if not found == True: ngl = dict(map(lambda (k,v): (k, abs(v-1)), ngl.iteritems()))
                    # np : next precision, i.e., p_{k+1}
                    #np = precisionSynth(model,dataset,image0,t,cl,gl,ncl,ngl,cp)
                    (ncl,ngl,np) = precisionSynth(t,ncl,ngl)
                    #print "the precision is %s."%(np)
                    
                    print "dimensions to be considered: %s"%(ncl)
                    #print "dimensions that have been considered before: %s"%(st.manipulated[t])
                    print "spans for the dimensions: %s"%(ngl)
                
                    #print ncl,ngl

                    if t == k: 
                        print "3) safety analysis ..."
                        # wk for the set of counterexamples
                        # rk for the set of images that need to be considered in the next precision
                        # rs remembers how many input images have been processed in the last round
                        # ncl and ngl are revised by considering the precision np
                        (ncl,ngl,rs,wk,rk) = safety_analysis(model,dataset,statisticsFile,t,startIndex,st,index,ncl,ngl,np)

                        print "4) add new images ..."
                        random.seed(time.time())
                        if len(rk) > numForEachDist: 
                            rk = random.sample(rk, numForEachDist)
                        diffs = diffImage(image0,rk[0])
                        print("the dimensions of the images that are changed in the previous round: %s"%diffs)
                        if len(diffs) == 0: st.clearManipulated(k)
                        st.addImages(model,rk)
                        st.removeProcessed(imageIndex)
                        (re,percent) = reportInfo(statisticsFile,image,rs,wk,mfn,howfar,image0)
                        break
                    else: 
                        print "3) add new intermediate node ..."
                        index = st.addIntermediateNode(image0,ncl,ngl,np,mfn,index)
                        re = False
                        t += 1
                if re == True: 
                    dc.addManipulationPercentage(percent)
                    (ocl,ocf) = NN.predictWithImage(model,rk[0])
                    dc.addConfidence(ocf)
                    break
                
            st.destructor()
        k += 1    
     
        runningTime = time.time() - start_time   
        dc.addRunningTime(runningTime)
        
    dc.outputOneSample()
    

def reportInfo(statisticsFile,image,rs,wk,mfn,howfar,image0):

    # exit only when we find an adversarial example
    if wk == []:    
        print "no adversarial example is found in this layer."  
        return (False,0)
    else: 
        print "an adversarial example has been found."
        diffs = diffImage(image,image0)
        elts = len(diffs.keys())
        if len(image0.shape) == 2: 
            percent = elts / float(len(image0)*len(image0[0]))
        elif len(image0.shape) == 1:
            percent = elts / float(len(image0))
        elif len(image0.shape) == 3:
            percent = elts / float(len(image0)*len(image0[0])*len(image0[0][0]))
        string = "%s input dimensions have been changed, i.e.,\n"%(elts)
        string += "%s of the input dimensions have been changed. \n"%(percent)
        print(string)
        statisticsFile.write(string)
        return (True,percent)
        
if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    