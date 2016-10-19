#!/usr/bin/env python

"""
author: Xiaowei Huang
"""

def maxpooling_safety_solve(input,activations):  

    if len(activations.shape) == 3: 
        for k in range(0,len(activations)): 
            for i in range(0,len(activations[0])):
                for j in range(0,len(activations[0][0])): 
                    input[k][2*i][2*j] = activations[k][i][j]
        return input
    else: 
        print("input shape %s has not been considered. "%(input.shape))

     