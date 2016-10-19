#!/usr/bin/env python

"""
author: Xiaowei Huang
"""


def flatten_safety_solve(input,activations):  

    if len(input.shape) == 3: 
        n = 0
        l = 0 
        for k in range(0,len(input)): 
            for i in range(0,len(input[0])):
                for j in range(0,len(input[0][0])): 
                    if input[k][i][j] != activations[l]: n += 1
                    input[k][i][j] = activations[l]
                    l += 1
        #print("%s dimension are different."%n)
        return input
    else: 
        print("input shape %s has not been considered. "%(input.shape))

     