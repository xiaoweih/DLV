#!/usr/bin/env python

import numpy as np
import time
import copy
import math
import collections
import random

import matplotlib.pyplot as plt
from scipy import ndimage



from conv_safety_solve import conv_safety_solve
from conv_adversarial_solve import conv_adversarial_solve

from dense_adversarial_solve import dense_adversarial_solve

import conv_bp
import relu

from basics import equalActivations, assure_path_exists

from configuration import * 


def adversarial_analysis(model,dataset,layer2Consider,whichIndex,cl,gl,pk):

    config = NN.getConfig(model)
        
    # get an image to interpolate
    image = NN.getImage(model,whichIndex)
    # get weights and bias of the entire trained neural network
    (wv,bv) = NN.getWeightVector(model,layer2Consider)

    # save original image
    originalImage = copy.deepcopy(image)
    # variable used to save last image
    lastImage = copy.deepcopy(image)

    # save the starting layer 
    originalLayer2Consider = copy.deepcopy(layer2Consider)

    #mm.show(originalImage)
    
    # predict with neural network
    (originalClass,originalConfident) = NN.predictWithImage(model,image)
    #dataBasics.save(image, directory_pic_string+"/"+str(whichIndex)+"_original_as_"+str(originalClass)+"_with_confidence_"+str(originalConfident)+".png")

    print "the right class is " + (str(dataBasics.LABELS(int(originalClass))))
    print "the confidence is " + (str(originalConfident))

    dataBasics.save(layer2Consider,image,directory_pic_string+"/temp.png")
    print "please refer to the file "+directory_pic_string+"/temp.png for the image under processing ..."
    
    # variable used to save the class for intermediate image
    newClass = originalClass
    
    # variable used to save the confident for intermediate image
    confident = copy.deepcopy(originalConfident)
    
    round = 1
    # FIXME: use nRound as the exit condition for now
    nRound = 100
    
    # 0:solve, 1: back-propagation
    whichAlgorithm = 0
    
    # read the size from the configuration
    csize = size
    # save the original size to start with 
    originalsize = copy.deepcopy(csize)
    
    # initialize t_k
    # as two sets: for precisions and for f_k
    tkp = [NN.getActivationValue(model,layer2Consider,image)]
    tkf = []
    wk = []
    
    # decide if need to use heuristic 
    usingHeuristics = True
    
    # a recursive procedure until find an interpolated image which is classified 
    # differently with the original image 
    # note: there are some other exit mechanisms in the looping body
    while newClass == originalClass: 

        print "\nin round: " + str(round)
        print "layer: " + str(layer2Consider)
        print "%s points of precision have been studied."%(len(tkp))
        print "%s regions have been studied."%(len(tkf))

        # get the type of the current layer
        layerType = [ lt for (l,lt) in config if layer2Consider == l ]
        if len(layerType) > 0: layerType = layerType[0]
        else: print "cannot find the layerType"

        # save the confident value
        # it is expected that the confident value continuously decreases
        # FIXME: is this a good way to restart? 
        lastConfident = copy.copy(confident)

        # get the weights and bias for the current layer
        wv2Consider, bv2Consider = getWeight(wv,bv,layer2Consider)
                
        # call different solving approaches according to 
        # the type of the layer and the type of the algorithm
        # FIXME: need to expand this to work with other cases, e.g., MaxPooling2D, Convolution3D
        if layerType == "Convolution2D" and whichAlgorithm == 0:  
            print "convolutional layer, solving ..."
            activations1 = NN.getActivationValue(model,layer2Consider,image)
            if layer2Consider == 0:
                activations0 = lastImage
            else: activations0 = NN.getActivationValue(model,layer2Consider-1,image)
            string = assure_path_exists(directory_pic_string+"/"+str(whichIndex)+"_original_as_"+str(originalClass))
            # bl to denote whether newInput is found
            # newInput is the activations of k-1 layer
            # checkedInput is the activations of k layer
            # regionInfo is a region that checkedInput is found by heuristics
            (bl,newInput,checkedInput,regionInfo) = conv_solve_prep(model,dataBasics,string,cl,gl,pk,tkp,tkf,activations0,wv2Consider,bv2Consider,activations1,csize,usingHeuristics)
            # if this is the layer on which the safety is checked
            #    and we found a point 
            tkf = tkf + regionInfo
            if (originalLayer2Consider == layer2Consider) and (bl == True) :  
                newPoint = True
                for i in range(len(tkp)): 
                    if equalActivations(checkedInput, tkp[i], pk):
                        newPoint = False
                        break
                if newPoint == True : tkp.append(checkedInput)
                whichAlgorithm = 1
            
        elif layerType == "Convolution2D" and whichAlgorithm == 1:
            print "convolutional layer, back-propagating ..."  
            if layer2Consider == 0 : 
                activations0 = lastImage
            else: activations0 = NN.getActivationValue(model,layer2Consider-1,lastImage)
            string = directory_pic_string+"/"+str(whichIndex)+"_original_as_"+str(originalClass)
            (bl,newInput,checkedInput,regionInfo) = conv_solve_prep(model,dataBasics,string,cl,gl,pk,tkp,tkf,activations0,wv2Consider,bv2Consider,image,originalsize,usingHeuristics)
    
        if layerType == "Dense" and whichAlgorithm == 0:  
            print "dense layer, solving ... %s"%(str(image))
            plt.plot([image[0]], [image[1]], 'g.')
            activations1 = NN.getActivationValue(model,layer2Consider,image)
            if layer2Consider == 0:
                activations0 = lastImage
            else: activations0 = NN.getActivationValue(model,layer2Consider-1,image)
            string = assure_path_exists(directory_pic_string+"/"+str(whichIndex)+"_original_as_"+str(originalClass))
            # bl to denote whether newInput is found
            # newInput is the activations of k-1 layer
            # checkedInput is the activations of k layer
            # regionInfo is a region that checkedInput is found by heuristics
            (bl,newInput,checkedInput,regionInfo) = dense_solve_prep(model,dataBasics,string,cl,gl,pk,tkp,tkf,activations0,wv2Consider,bv2Consider,activations1,csize,usingHeuristics,whichAlgorithm)
            whichAlgorithm = 1
            
        elif layerType == "Dense" and whichAlgorithm == 1:  
            print "dense layer, back propogation ... "
            if layer2Consider == 0 : 
                activations0 = lastImage
            else: activations0 = NN.getActivationValue(model,layer2Consider-1,lastImage)
            string = directory_pic_string+"/"+str(whichIndex)+"_original_as_"+str(originalClass)
            (bl,newInput,checkedInput,regionInfo) = dense_solve_prep(model,dataBasics,string,cl,gl,pk,tkp,tkf,activations0,wv2Consider,bv2Consider,image,csize,usingHeuristics,whichAlgorithm)
    
        elif layerType == "relu" and whichAlgorithm == 0:
            print "relu layer, solving ... (FIXME: do nothing at the moment)"
            activations1 = NN.getActivationValue(model,layer2Consider,image)
            activations0 = NN.getActivationValue(model,layer2Consider-1,image)
            (bl,newInput) = solve_relu(model,activations0,wv2Consider,bv2Consider,activations1)
            whichAlgorithm = 1
            
        elif layerType == "InputLayer" and whichAlgorithm == 1:
            print "input layer, back-propagating ..."
            (bl,newInput) = (True, image)
    
        elif layerType == "relu" and whichAlgorithm == 1:
            print "relu layer, back-propagating ..."
            activations0 = NN.getActivationValue(model,layer2Consider-1,lastImage)
            (bl,newInput) = relu.bp(activations0,image)
            #dataBasics.save(newInput[6],directory_pic_string+"/temp/"+str(round)+"temp"+str(layer2Consider)+".png")

        elif layerType == "ZeroPadding2D" and whichAlgorithm == 0:
            print "ZeroPadding2D layer, solving ... "
            activations1 = NN.getActivationValue(model,layer2Consider,lastImage)
            image1 = NN.removeZeroPadding2D(activations1)
            (bl,newInput) = (True,image1)
            whichAlgorithm = 1
            
        elif layerType == "ZeroPadding2D" and whichAlgorithm == 1:
            print "ZeroPadding2D layer, solving ... "
            image1 = NN.removeZeroPadding2D(image)
            (bl,newInput) = (True,image1)
            whichAlgorithm = 1

        # decide the next step according to the results from the solving
        if bl == False:   
            # if back-propagation fails    
            print "back-propagation or solving fails ..." 
            
            # if checking the safety 
            if task == "safety_check": 
                # if we cannot find a valid point
                #     by using taking heuristics
                if (originalLayer2Consider == layer2Consider) and (usingHeuristics == True) : 
                    # add the region covered by heuristics 
                    print("jumping to another region ... ")
                    usingHeuristics = False
                elif not (originalLayer2Consider == layer2Consider) : 
                    whichAlgorithm = 0
                    layer2Consider = copy.deepcopy(originalLayer2Consider)
                    image = newInput
                    lastImage = copy.deepcopy(image)
                #     by considering the entire region e_k
                else: 
                    print("the region of layer %s has been fully explored. "%(originalLayer2Consider))
                    print("we found %s adversarial examples."%(len(wk)))
                    break
            # to find an adversarial example
            elif task == "adversary_generation":  
                layer2Consider = copy.deepcopy(originalLayer2Consider)
                image = copy.deepcopy(originalImage)
                whichAlgorithm = 0
                csize = copy.deepcopy(originalsize)    
            round += 1
                
        elif layer2Consider > 0:       
            # still not yet reach the input layer
            # continue back-propagating 
            layer2Consider -= 1
            image = newInput
            # using heuristics when there is a point found in the last round (i.e., bl == True)
            usingHeuristics = True
            
        else:         
            # reached the input layer
            # check to see if the new input is classified wrongly.
            
            # FIXME: use the following code if you want to 
            # get prediction value by the obtained image 
            (newClass,confident) = NN.predictWithImage(model,newInput)
            path2 = directory_pic_string+"/temp_(round="+str(round)+").png"
            dataBasics.save(layer2Consider,newInput, path2)
            plt.plot([newInput[0]], [newInput[1]], 'g.')

            #newInput_forDoubleCheck = NN.readImage(path2)
            #(newClass_forDoubleCheck,confident_forDoubleCheck) = NN.predictWithImage(model,newInput_forDoubleCheck)
            #if newClass_forDoubleCheck != newClass: print("double checking fails. ")
            #else: print("double checking passed. ")
            
            # this is for testing purpose
            # to double check the class for the original image 
            (newClass_forTest,confident_forTest) = NN.predictWithImage(model,originalImage)
            
            difference = newInput - originalImage
            
            if newClass_forTest != originalClass: print "ERROR: something must be wrong ... "
            
            print "confident level: " + str(confident)
            # Great! we found an image which has different class with the original image
            if newClass != originalClass: 
                newClassStr = dataBasics.LABELS(int(newClass))
                origClassStr = dataBasics.LABELS(int(originalClass))
                print "Class changed! from " + str(origClassStr) +" into " + str(newClassStr)
                
                path0= directory_pic_string+"/"+str(whichIndex)+"_original_as_"+str(origClassStr)+"_with_confidence_"+str(originalConfident)+".png"
                dataBasics.save(layer2Consider,originalImage, path0)

                path1 = directory_pic_string+"/"+str(whichIndex)+"_modified_into_"+str(newClassStr)+"_with_confidence_"+str(confident)+".png"
                dataBasics.save(layer2Consider,newInput, path1)
                
                # add a point whose class is wrong
                wk.append(NN.getActivationValue(model,originalLayer2Consider,newInput))
                
                #dataBasics.save(difference, directory_pic_string+"/"+str(whichIndex)+"_difference.png")
                return True
            # we have tried too hard. Let's say to rest for now ...
            elif round > nRound: 
                print "have tried for " + str(nRound) + " rounds and did not find any counterexample ... "
                plt.savefig(directory_pic_string+"/temp_(round=-1).png")
                return False
            # increase the size with respect to the confident level 
            elif confident >= lastConfident and csize < maxsize: # and csize < len(originalImage[0]) -3 :
                # if the confidence level does not decrease: 
                #    increase the change rate
                csize += step
                print "confidence level does not decrease ... "
                print "change size of the study region to " + str(csize)
                #dataBasics.save(newInput, directory_pic_string+"/temp/"+str(whichIndex)+"_modified_into_"+str(newClass)+"_with_confidence_"+str(confident)+".png")

            # the size has covered the entire image
            # FIXME: shall we stop or let it run until reaching the rounds bound
            #elif csize >= len(originalImage[0]) -3: 
            #    print "reach the limit of the image ... "
            #    return False
                #print "size stay the same. "
                
            # FIXME: why not put this with one of the above cases? 
            # in case when we want to continue for a next round
            # update the image into the one obtained in this round
            whichAlgorithm = 0
            layer2Consider = copy.deepcopy(originalLayer2Consider)
            image = newInput
            lastImage = copy.deepcopy(image)
            round += 1
            
            # using heuristics when there is a point found in the last round (i.e., bl == True)
            usingHeuristics = True
            
    return False

#####################################################################################
#
#  for solving relu layer
#  have not implemented 
#######################################################################################

def solve_relu(model,input,wv,bv,activations):
    # FIXME: 
    return (True,input)


#######################################################################################
#
# preparation for solving convolutional layer
#
#####################################################################################

def conv_solve_prep(model,dataBasics,string,cl,gl,pk,tkp,tkf,input,wv,bv,activations,csize,usingHeuristics):

    # filters can be seen as the output of a convolutional layer
    nfilters = numberOfFilters(wv)
    # features can be seen as the inputs for a convolutional layer
    nfeatures = numberOfFeatures(wv)

    # space holders for computation values
    biasCollection = {}
    filterCollection = {}
    
    for l in range(nfeatures):
        for k in range(nfilters):
            filter = [ w for ((p1,c1),(p,c),w) in wv if c1 == l+1 and c == k+1 ]
            bias = [ w for (p,c,w) in bv if c == k+1 ]
            if len(filter) == 0 or len(bias) == 0 : 
                print "error: bias =" + str(bias) + "\n filter = " + str(filter)
            else:
                filter = filter[0]
                bias = bias[0]
                    
            # flip the filter for convolve
            flipedFilter = np.fliplr(np.flipud(filter))
            biasCollection[l,k] = bias
            filterCollection[l,k] = flipedFilter
            #print filter.shape
            
    
    #points = getProcessPoints(nfeatures,input,size)
    input2 = copy.deepcopy(input)
    ri = []
    #for i in range(len(points)):
    point = getRandomPoint(nfeatures,input,csize) # points[i]
    print("processing point "+str(point))
    #FIXME: 
    
    (bl1,newInput,checkedInput,regionInfo) = conv_adversarial_solve(nfeatures,nfilters,filterCollection,biasCollection,input,activations,point)
        
    #dataBasics.save(input2, string+"_"+str(point)+".png")
    #print("bl1="+str(bl1)+"   newInput="+str(len(newInput)))
    if bl1 == False: ri.append(regionInfo)
    print("completed a round of processing of the entire image ")
    return (bl1,newInput,checkedInput,ri)
    
    
#######################################################################################
#
# preparation for solving dense layer
#
#####################################################################################

def dense_solve_prep(model,dataBasics,string,cl,gl,pk,tkp,tkf,input,wv,bv,activations,csize,usingHeuristics,whichAlgorithm):

    # filters can be seen as the output of a convolutional layer
    nfilters = numberOfFilters(wv)
    # features can be seen as the inputs for a convolutional layer
    nfeatures = numberOfFeatures(wv)

    # space holders for computation values
    biasCollection = {}
    filterCollection = {}
    
    for l in range(nfeatures):
        for k in range(nfilters):
            filter = [ w for ((p1,c1),(p,c),w) in wv if c1 == l+1 and c == k+1 ]
            bias = [ w for (p,c,w) in bv if c == k+1 ]
            if len(filter) == 0 or len(bias) == 0 : 
                print "error: bias =" + str(bias) + "\n filter = " + str(filter)
            else:
                filter = filter[0]
                bias = bias[0]
                    
            biasCollection[l,k] = bias
            filterCollection[l,k] = filter
    
    if whichAlgorithm == 0: 
        (bl1,newInput,checkedInput,regionInfo) = dense_adversarial_solve(nfeatures,nfilters,filterCollection,biasCollection,input,activations)
    else: 
        (bl1,newInput,checkedInput,regionInfo) = dense_adversarial_solve(nfeatures,nfilters,filterCollection,biasCollection,input,activations)
                
    print("completed a round of processing ")
    return (bl1,newInput,checkedInput,regionInfo)
    
compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
    
###################
### for back-propagation, not used at the moment
#################

def conv_bp_prep(model,input,wv,bv,activations,csize):

    nfilters = numberOfFilters(wv)
    nfeatures = numberOfFeatures(wv)
    
    print "number of filters: " + str(nfilters)
    print "number of features in the previous layer: " + str(nfeatures)
    
    (_, sizex, sizey) = activations.shape
    sizex += 2

    # space holders for computation values
    biasCollection = {}
    filterCollection = {}
    
    for l in range(nfeatures):
        for k in range(nfilters):
            filter = [ w for ((p1,c1),(p,c),w) in wv if c1 == l+1 and c == k+1 ]
            bias = [ w for (p,c,w) in bv if c == k+1 ]
            if len(filter) == 0 or len(bias) == 0 : 
                print "error: bias =" + str(bias) + "\n filter = " + str(filter)
            else:
                filter = filter[0]
                bias = bias[0]
                    
            flipedFilter = np.fliplr(np.flipud(filter))
            biasCollection[l,k] = bias
            filterCollection[l,k] = flipedFilter
            #print filter.shape
    
    
    (bl,newInput) = conv_bp.bp(nfeatures,nfilters,filterCollection,biasCollection,input,activations,csize)
    
    return (bl,newInput)


def getProcessPoints(nfeatures,input,csize):

    if nfeatures == 1: images = np.expand_dims(input, axis=0)
    else: images = input

    d = 1

    x = len(images[0])
    y = len(images[0][0])
    mx = int(math.floor(x/2))
    my = int(math.floor(y/2))
    
    if x > 100 : 
        num = x / (csize * d)
    else: num = x / (csize * 2 * d)
    
    lu = [(mx - d * csize * k1 , my - d * csize * k2) for  k1 in range(num) for  k2 in range(num) if mx - d * csize * k1 >= csize and my - d * csize * k2 >= csize ]
    ld = [(mx + d * csize * k1 , my - d * csize * k2) for  k1 in range(num) for  k2 in range(num) if mx + d * csize * k1 <= x - csize and my - d * csize * k2 >= csize ]
    ru = [(mx - d * csize * k1 , my + d * csize * k2) for  k1 in range(num) for  k2 in range(num) if mx - d * csize * k1 >= csize and my + d * csize * k2 <= y - csize ]
    rd = [(mx + d * csize * k1 , my + d * csize * k2) for  k1 in range(num) for  k2 in range(num) if mx + d * csize * k1 <= x - csize and my + d * csize * k2 <= y - csize ]
    lm = [(mx,my)]

    return list(set(lu+ld+ru+rd+lm))
    
    
def getRandomPoint(nfeatures,input,csize):

    if nfeatures == 1: images = np.expand_dims(input, axis=0)
    else: images = input
    
    x = len(images[0])
    y = len(images[0][0])
    
    xs = [ p + csize + filterSize - 1 for p in range(x-2*csize-2*(filterSize-1))]
    ys = [ p + csize + filterSize - 1 for p in range(x-2*csize-2*(filterSize-1))]
    
    return (random.choice(xs), random.choice(ys))
    
def getWeight(wv,bv,layerIndex):
    wv = [ (a,(p,c),w) for (a,(p,c),w) in wv if p == layerIndex ]
    bv = [ (p,c,w) for (p,c,w) in bv if p == layerIndex ]
    return (wv,bv)
    
def numberOfFilters(wv):
    return np.amax((zip (*((zip (*wv))[1])))[1])

#  the features of the last layer
def numberOfFeatures(wv):
    return np.amax((zip (*((zip (*wv))[0])))[1])
    
    
def imageSize(image):
    return len(image[0])
    

           
###################################
## the following is for testing purpose 
###################################

def manipulate(dataBasics,NN,image):

    dataBasics.save(-1,image,"temp2.png")



    image0 = copy.deepcopy(image)
    for i  in range(5): 
        image[0] = scale(image[0],5)
        image[1] = scale(image[1],5)
        image[2] = scale(image[2],5)
    print(np.amax(image) == np.amax(image0))


    dataBasics.save(-1,image,"temp3.png")
    print("saved.")
    print(image.shape)
    
def scale(mat,num):
    max=[]
    min=[]
    for k in range(len(mat)):
        for j in range(len(mat[0])):
            if len(max) < num: 
                max.append([k,j,mat[k][j]])
                min.append([k,j,mat[k][j]])
            else: 
                for i in range(len(max)): 
                    if mat[max[i][0]][max[i][1]] > max[i][2]: 
                        max[i] = [k,j,mat[k][j]]
                for i in range(len(min)): 
                    if mat[min[i][0]][min[i][1]] < min[i][2]: 
                        min[i] = [k,j,mat[k][j]]
    for (i1,i2,n) in max:
        mat[i1][i2] = n - abs(n) * 0.5
    for (i1,i2,n) in min:
        mat[i1][i2] = n + abs(n) * 0.5  
    return mat  
    
def convolve(image,filter): 
    image = image.flatten()
    filter = filter.flatten()
    value = 0
    for i in range(len(image)):
        value += image[i] * filter[i]
    return value


def testModel(model,image):
        # FIXME: testing the mathematics between layers 

    # get weights and bias of the entire trained neural network
    (wv,bv) = NN.getWeightVector(model)
    
    activations1 = NN.getActivationValue(model,0,image)
    activations2 = NN.getActivationValue(model,1,image)
    #print wv2Consider
    #print bv2Consider    
    
    print activations1[0] 
    print np.min(activations2)
    

        
"""
for testing of convolutional layers

    (image3, activations3) = NN.getActivationValue(model,2,whichIndex)
    (image2, activations2) = NN.getActivationValue(model,1,whichIndex)
    wv2Consider, bv2Consider = getWeight(wv,bv,2)
    #print wv2Consider
    #print bv2Consider    
    
    (nfeatures,row,col) = activations2.shape
    nfilters = numberOfFilters(wv2Consider) 
    nfeatures = numberOfFeatures(wv2Consider)

    # space holders for computation values
    value1 = np.zeros( (nfilters,row-2,col-2) )
    value2 = np.zeros( (nfilters,row-2,col-2) )

    biasCollection = {}
    filterCollection = {}
    

    for k in range(nfilters):
        bias = [ w for (p,c,w) in bv2Consider if c == k+1 ]
        bias = bias[0]
        biasCollection[k] = bias
        act3 = np.squeeze(activations3[k:k+1])
        for i in range(row-2):
            for j in range(col-2):
                value1[k][i][j] = bias
                value2[k][i][j] = bias
                for l in range(nfeatures):
                    act2 = np.squeeze(activations2[l:l+1])

                    sample = act2[i:i+3,j:j+3]
                    filter = [ w for ((p1,c1),(p,c),w) in wv2Consider if c1 == l+1 and c == k+1 ]
                    if len(filter) == 0: 
                        print "error: no filters "
                    else:
                        filter = filter[0]
                        
                    #print "k="+str(k)+"  l="+str(l)+"  i="+str(i)+"  j="+str(j)
                    #print "sample"+str(sample.shape)
                    #print filter.shape
                    convolved = ndimage.convolve(sample, filter, mode='constant', cval=0.0)
                    value1[k][i][j] += convolved[1][1]
                    flipedFilter = np.fliplr(np.flipud(filter))
                    value2[k][i][j] += convolve(sample,flipedFilter) 
                    
                    
                    filterCollection[l,k] = flipedFilter
                    #print filter.shape

                print "value1=" + str(value1[k][i][j])+"   value2=" + str(value2[k][i][j]) + "   activations="+str(np.squeeze(act3[i:i+1,j:j+1]))+"  distance="+str((value1[k][i][j]-np.squeeze(act3[i:i+1,j:j+1]))/bias)
"""