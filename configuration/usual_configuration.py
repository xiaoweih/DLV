#!/usr/bin/env python

"""
Define paramters
author: Xiaowei Huang
"""

def usual_configuration(dataset):

    if dataset == "twoDcurve": 
    
        # which image to start with or work with 
        # from the database
        startIndexOfImage = 0
        
        # the start layer to work from 
        startLayer = 0

        # the maximal layer to work until 
        maxilayer = 1

        ## number of features of each layer 
        # in the paper, dims_L = numOfFeatures * featureDims
        numOfFeatures = 0

        # use linear restrictions or conv filter restriction
        inverseFunction = "point"
        #inverseFunction = "area"

        # point-based or line-based, or only work with a specific point
        #enumerationMethod = "convex"
        enumerationMethod = "line"

        # heuristics for deciding a region
        heuristics = "Activation"
        #heuristics = "Derivative"

        # do we need to repeatedly select an updated input neuron
        repeatedManipulation = "allowed"
        #repeatedManipulation = "disallowed"

        #checkingMode = "specificLayer"
        checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        exitWhen = "foundAll"
        #exitWhen = "foundFirst"
        
        # do we need to generate temp_.png files
        #tempFile = "enabled"
        tempFile = "disabled"
        
        # compute the derivatives up to a specific layer
        derivativelayerUpTo = 3

        return (startIndexOfImage,startLayer,maxilayer,numOfFeatures,inverseFunction,enumerationMethod,heuristics,repeatedManipulation,checkingMode,exitWhen,derivativelayerUpTo,tempFile)
        
    elif dataset == "mnist": 

        # which image to start with or work with 
        # from the database
        startIndexOfImage = 2448
        
        # the start layer to work from 
        startLayer = 0

        # the maximal layer to work until 
        maxilayer = 0

        ## number of features of each layer 
        numOfFeatures = 150

        # use linear restrictions or conv filter restriction
        inverseFunction = "point"
        #inverseFunction = "area"

        # point-based or line-based, or only work with a specific point
        enumerationMethod = "convex"
        #enumerationMethod = "line"

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
        
        # do we need to generate temp_.png files
        #tempFile = "enabled"
        tempFile = "disabled"
        
        # compute the derivatives up to a specific layer
        derivativelayerUpTo = 3
    
        return (startIndexOfImage,startLayer,maxilayer,numOfFeatures,inverseFunction,enumerationMethod,heuristics,repeatedManipulation,checkingMode,exitWhen,derivativelayerUpTo,tempFile)
        
    elif dataset == "cifar10": 
    
        # which image to start with or work with 
        # from the database
        startIndexOfImage = 331
        
        # the start layer to work from 
        startLayer = 0

        # the maximal layer to work until 
        maxilayer = 0

        ## number of features of each layer 
        numOfFeatures = 500

        # use linear restrictions or conv filter restriction
        inverseFunction = "point"
        #inverseFunction = "area"

        # point-based or line-based, or only work with a specific point
        enumerationMethod = "convex"
        #enumerationMethod = "line"

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
        
        # do we need to generate temp_.png files
        #tempFile = "enabled"
        tempFile = "disabled"
        
        # compute the derivatives up to a specific layer
        derivativelayerUpTo = 5
    
        return (startIndexOfImage,startLayer,maxilayer,numOfFeatures,inverseFunction,enumerationMethod,heuristics,repeatedManipulation,checkingMode,exitWhen,derivativelayerUpTo,tempFile)

    elif dataset == "imageNet": 
    
        # which image to start with or work with 
        # from the database
        startIndexOfImage = 3
        
        # the start layer to work from 
        startLayer = 0

        # the maximal layer to work until 
        maxilayer = 1

        ## number of features of each layer 
        numOfFeatures = 20000

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

        checkingMode = "specificLayer"
        #checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        #exitWhen = "foundAll"
        exitWhen = "foundFirst"
        
        # do we need to generate temp_.png files
        tempFile = "enabled"
        #tempFile = "disabled"
        
        # compute the derivatives up to a specific layer
        derivativelayerUpTo = 5
    
        return (startIndexOfImage,startLayer,maxilayer,numOfFeatures,inverseFunction,enumerationMethod,heuristics,repeatedManipulation,checkingMode,exitWhen,derivativelayerUpTo,tempFile)