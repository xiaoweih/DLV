#!/usr/bin/env python

"""
Define paramters
"""
import numpy as np
import os

import mnist_network as NN_mnist
import cifar10_network as NN_cifar10
import imageNet_network as NN_imageNet
import regression_network as NN_regression


import mnist
import cifar10
import imageNet
import regression


def network_parameters(dataset): 

    
#######################################################
#
#  Bound for the elements of input 
#
#######################################################

    if dataset in ["mnist","cifar10","imageNet"] : 
        bound = [0,1]
    elif dataset == "regression": 
        bound = [0, 2 * np.pi]
    else: 
        bound = [0,0]

#######################################################
#
#  some auxiliary parameters that are used in several files
#  they can be seen as global parameters for an execution
#
#######################################################
    


# which dataset to analyse
    if dataset == "mnist": 
        NN = NN_mnist
        dataBasics = mnist
        directory_model_string = makedirectory("networks/mnist")
        directory_statistics_string = makedirectory("data/mnist_statistics")
        directory_pic_string = makedirectory("data/mnist_pic")
        
# ce: the region definition for layer 0, i.e., e_0
        featureDims = 2
        span = 255/float(255)
        numSpan = 1
# cp : current precision, i.e., p_k
        precision = 255/float(255)
    
    elif dataset == "regression": 
        NN = NN_regression
        dataBasics = regression
        directory_model_string = makedirectory("networks/regression")
        directory_statistics_string = makedirectory("data/regression_statistics")
        directory_pic_string = makedirectory("data/regression_pic")

# ce: the region definition for layer 0, i.e., e_0
        featureDims = 5
        span = 255/float(255)
        numSpan = 1
# cp : current precision, i.e., p_k
        precision = 255/float(255)

    elif dataset == "cifar10": 
        NN = NN_cifar10
        dataBasics = cifar10
        directory_model_string = makedirectory("networks/cifar10")
        directory_statistics_string = makedirectory("data/cifar10_statistics")
        directory_pic_string = makedirectory("data/cifar10_pic")
 
# ce: the region definition for layer 0, i.e., e_0
        featureDims = 5
        span = 255/float(255)
        numSpan = 1
# cp : current precision, i.e., p_k
        precision = 255/float(255)
                
    elif dataset == "imageNet": 
        NN = NN_imageNet
        dataBasics = imageNet
        directory_model_string = makedirectory("networks/imageNet")
        directory_statistics_string = makedirectory("data/imageNet_statistics")
        directory_pic_string = makedirectory("data/imageNet_pic")

# ce: the region definition for layer 0, i.e., e_0
        featureDims = 5
        span = 125
        numSpan = 1
# cp : current precision, i.e., p_k
        precision = 250
            
#######################################################
#
#  size of the filter used in convolutional layers
#
#######################################################
    
    filterSize = 3 

    return (featureDims,span,numSpan,precision,bound,NN,dataBasics,directory_model_string,directory_statistics_string,directory_pic_string,filterSize)

def makedirectory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    return directory_name