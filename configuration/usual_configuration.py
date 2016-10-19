#!/usr/bin/env python

"""
Define paramters
"""

def usual_configuration(dataset):

    if dataset == "twoDcurve": 
    
        # which image to start with or work with 
        # from the database
        startIndex = 0

        # the maximal layer to work until 
        maxilayer = 2

        ## number of features of each layer 
        # in the paper, dims_L = numOfFeatures * featureDims
        numOfFeatures = 0

        # use linear restrictions or conv filter restriction
        feedback = "point"
        #feedback = "area"

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

        return (startIndex,maxilayer,numOfFeatures,feedback,enumerationMethod,heuristics,repeatedManipulation,checkingMode,exitWhen,derivativelayerUpTo,tempFile)
        
    elif dataset == "mnist": 

        # which image to start with or work with 
        # from the database
        startIndex = 2438

        # the maximal layer to work until 
        maxilayer = 0

        ## number of features of each layer 
        numOfFeatures = 1000

        # use linear restrictions or conv filter restriction
        feedback = "point"
        #feedback = "area"

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
    
        return (startIndex,maxilayer,numOfFeatures,feedback,enumerationMethod,heuristics,repeatedManipulation,checkingMode,exitWhen,derivativelayerUpTo,tempFile)
        
    elif dataset == "cifar10": 
    
        # which image to start with or work with 
        # from the database
        startIndex = 330

        # the maximal layer to work until 
        maxilayer = 0

        ## number of features of each layer 
        numOfFeatures = 1000

        # use linear restrictions or conv filter restriction
        feedback = "point"
        #feedback = "area"

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
    
        return (startIndex,maxilayer,numOfFeatures,feedback,enumerationMethod,heuristics,repeatedManipulation,checkingMode,exitWhen,derivativelayerUpTo,tempFile)

    elif dataset == "imageNet": 
    
        # which image to start with or work with 
        # from the database
        startIndex = 0

        # the maximal layer to work until 
        maxilayer = 1

        ## number of features of each layer 
        numOfFeatures = 20000

        # use linear restrictions or conv filter restriction
        feedback = "point"
        #feedback = "area"

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
    
        return (startIndex,maxilayer,numOfFeatures,feedback,enumerationMethod,heuristics,repeatedManipulation,checkingMode,exitWhen,derivativelayerUpTo,tempFile)