#!/usr/bin/env python


from configuration import *


def getLayerType(model,layer2Consider):

    config = NN.getConfig(model)

    # get the type of the current layer
    layerType = [ lt for (l,lt) in config if layer2Consider == l ]
    if len(layerType) > 0: layerType = layerType[0]
    else: print "cannot find the layerType"
    
    return layerType
    
