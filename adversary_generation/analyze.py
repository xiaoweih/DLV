#!/usr/bin/env python


import numpy as np

import h5py

import display
import transform_NN as tnn
import solve
import basics

import analyze_SMT
import analyze_sampling


def analyze(model0,x,y,xbounds,ybounds,plt):

    #return analyze_SMT.analyze(model0,mapping,x,y,xbounds,ybounds,plt)

    return analyze_sampling.analyze(model0,x,y,xbounds,ybounds,plt)

