#!/usr/bin/env python

"""
Define paramters
author: Xiaowei Huang
"""

from network_configuration import *

from usual_configuration import * 


#######################################################
#
#  To find counterexample or do safety checking
#
#######################################################

task = "safety_check"

#######################################################
#
#  The following are parameters to indicate how to work 
#   with a problem
#
#######################################################

# which dataset to work with
#dataset = "twoDcurve"
#dataset = "mnist"
dataset = "cifar10"
#dataset = "imageNet"

# decide whether to take an experimental configuration
# for specific dataset
experimental_config = True
#experimental_config = False

# the network is trained from scratch
#  or read from the saved files
whichMode = "read"
#whichMode = "train"

# work with a single image or a batch of images 
#dataProcessing = "single"
dataProcessing = "batch"
dataProcessingBatchNum = 3

#######################################################
#
#  1. parameters related to the networks
#
#######################################################

span = 255/float(255)   # s_p in the paper
numSpan = 1.0          # m_p in the paper
featureDims = 5         # dims_{k,f} in the paper

# error bounds, defaulted to be 1.0
# \varepsilon in the paper
errorBounds = {}
errorBounds[-1] = 1.0

#######################################################
#  get parameters from network_configuration
#######################################################

(featureDims,span,numSpan,errorBounds,boundOfPixelValue,NN,dataBasics,directory_model_string,directory_statistics_string,directory_pic_string,filterSize) = network_parameters(dataset)


#######################################################
#
#  2. The following are parameters for safety checking
#     only useful only when experimental_config = False
#
#######################################################

# which image to start with or work with 
# from the database
startIndexOfImage = 197

# the maximal layer to work until 
startLayer = 0

# the maximal layer to work until 
maxLayer = 3

## number of features of each layer 
# in the paper, dims_L = numOfFeatures * featureDims
numOfFeatures = 40

# use linear restrictions or conv filter restriction
inverseFunction = "point"
#inverseFunction = "area"

# point-based or line-based, or only work with a specific point
enumerationMethod = "convex"
#enumerationMethod = "line"
#enumerationMethod = "point"

# heuristics for deciding a region
heuristics = "Activation"
#heuristics = "Derivative"

# do we need to repeatedly select an updated input neuron
#repeatedManipulation = "allowed"
repeatedManipulation = "disallowed"

#checkingMode = "specificLayer"
checkingMode = "stepwise"

# exit whenever an adversarial example is found
#exitWhen = "foundAll"
exitWhen = "foundFirst"

# compute the derivatives up to a specific layer
derivativelayerUpTo = 3

# do we need to generate temp_.png files
#tempFile = "enabled"
tempFile = "disabled"


#######################################################
#  get parameters for the case when experimental_config = True
#######################################################

if experimental_config == True: 
    (startIndexOfImage,startLayer, maxLayer,numOfFeatures,inverseFunction,enumerationMethod,heuristics,repeatedManipulation,checkingMode,exitWhen,derivativelayerUpTo,tempFile) = usual_configuration(dataset)
    

############################################################
#
#  3. other parameters that are believed to be shared among all cases
#  FIXME: check to see if they are really needed/used
#
################################################################

## reset percentage
#  applies when manipulated elements do not increase
reset = "onEqualManipulationSet"
#reset = "Never"

## how many branches to expand 
numOfPointsAfterEachFeature = 1

# impose bounds on the input or not
boundRestriction = True

# timeout for z3 to handle a run
timeout = 600



############################################################
#
#  some miscellaneous parameters 
#   which need to confirm whether they are useful
#  FIXME: check to see if they are really needed/used
#
################################################################


# how many pixels per feature will be changed 
num = 3 #csize - 3

# for conv_solve_prep 
# the size of the region to be modified
#if imageSize(originalImage) < 50: 
size = 4
maxsize = 32
step = 0

# the error bound for manipulation refinement 
# between layers
epsilon = 0.1


############################################################
#
#  a parameter to decide whether 
#  FIXME: check to see if they are really needed/used
#
################################################################

# 1) the stretch is to decide a dimension of the next layer on 
#     the entire region of the current layer
# 2) the condense is to decide several (depends on refinementRate) dimensions 
#     of the next layer on a manipulation of a single dimension of the current layer

#regionSynthMethod = "stretch"
regionSynthMethod = "condense"

############################################################
#
#  a function to decide how many features to be manipulated in each layer
#  this function is put here for its related to the setting of an execution
#  FIXME: check to see if they are really needed/used
#
################################################################

# this parameter tells how many elements will be used to 
#  implement a manipulation from a single element of the previous layer
refinementRate = 1
    
def getManipulatedFeatureNumber(model,numDimsToMani,layer2Consider): 

    config = NN.getConfig(model)
    
    # get the type of the current layer
    layerType = [ lt for (l,lt) in config if layer2Consider == l ]
    if len(layerType) > 0: layerType = layerType[0]
    else: print "cannot find the layerType"

    if layerType == "Convolution2D":  
        return numDimsToMani # + 1
    elif layerType == "Dense":  
        return numDimsToMani * refinementRate
    else: return numDimsToMani
    
#######################################################
#
#  show detailedInformation or not
#  FIXME: check to see if they are really needed/used
#
#######################################################

detailedInformation = False

def nprint(str):
    if detailedInformation == True: 
        print(str)        
        
