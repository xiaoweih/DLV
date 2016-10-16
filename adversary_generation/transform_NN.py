#!/usr/bin/env python


import numpy as np
import copy

def transform(model0):

    model = copy.deepcopy(model0)
    for layer in model.layers:
    	 index=model.layers.index(layer)
         h=layer.get_weights()
         h1 = []
         if len(h) > 1 :
         	h0=transformWeightList(h[0])
         	h1.append(h0)
         	h1.append(h[1])
         else: h1 = h
         model.layers[index].set_weights(h1)
    return model
    
def transformWeightList(wlist): 

    wlist0 = wlist.tolist()
    wlist1 = []
    for w in wlist: 
       wlist1.append(transformWeight(w))
    return np.array(wlist1)
    
def transformWeight(weight): 

    weight1 = []
    for e in weight: 
        e1 = e+0.1
        weight1.append(e1)
    return weight1