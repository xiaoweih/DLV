#!/usr/bin/env python

"""
"""

import sys
import os
sys.path.append('networks')
sys.path.append('safety_check')
sys.path.append('configuration')

import time
import numpy as np
import copy 
import random
import matplotlib.pyplot as plt

from loadData import loadData
from configuration import * 
        
def main():


    #dataset = "regression"
    #dataset = "mnist"
    dataset = "cifar10"
    #dataset = "imageNet"
    

    filename1 = "197_original_as_ship_with_confidence_0.990640580654.png"
    filename2 = "temp0_(round=32).png"

    imageNetPath1 = "%s/%s"%(directory_pic_string,filename1)
    imageNetPath2 = "%s/%s"%(directory_pic_string,filename2)

    image1 = NN.readImage(imageNetPath1)
    image2 = NN.readImage(imageNetPath2)
    
    k = diffImage(image1,image2)
    print "%s input elements are changed."%(k)
    
    (model,dataset,maxilayer,startIndex) = loadData()

    (class1,confidence1) = NN.predictWithImage(model,image1)
    classStr1 = dataBasics.LABELS(int(class1))
    print "the class for the first image is %s (%s) with confidence %s"%(class1,classStr1,confidence1)
    
    (class2,confidence2) = NN.predictWithImage(model,image2)
    classStr2 = dataBasics.LABELS(int(class2))
    print "the class for the first image is %s (%s) with confidence %s"%(class2,classStr2,confidence2)

    return 0
    
    
def diffImage(image1,image2):
    k = 0
    if len(image1.shape) == 1: 
        for x in range(len(image1)):
                if image1[x] != image2[x]: 
                    k += 1
                    print("dimension %s is changed from %s to %s"%(x,image1[x],image2[x]))
    elif len(image1.shape) == 2:
        for x in range(len(image1)):
            for y in range(len(image1[0])):
                if image1[x][y] != image2[x][y]: 
                    k += 1
                    print("dimension (%s,%s) is changed from %s to %s"%(x,y,image1[x][y],image2[x][y]))
    elif len(image1.shape) == 3:
        for x in range(len(image1)):
            for y in range(len(image1[0])):
                for z in range(len(image1[0][0])):
                    if image1[x][y][z] != image2[x][y][z]: 
                        k += 1
                        print("dimension (%s,%s,%s) is changed from %s to %s"%(x,y,z,image1[x][y][z],image2[x][y][z]))
    return k

def makedirectory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    return directory_name

def readImageForImageNet(path):

    import cv2
    
    im = cv2.resize(cv2.imread(path), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    
    return np.squeeze(im)

        
if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    