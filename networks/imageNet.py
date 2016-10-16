import os, struct
from array import array as pyarray
from cvxopt.base import matrix
import numpy as np

import PIL.Image

# FIXME: need actual class names
def LABELS(index): 
    ls = labels()
    if len(ls) > 0: 
       return ls[index]
    else: return range(1000)[index] 

def labels():

    file = open('networks/imageNet/caffe_ilsvrc12/synset_words.txt', 'r')
    data = file.readlines()
    ls = []
    for line in data:
        words = line.split()
        ls.append(' '.join(words[1:]))
    return ls

def save(layer,image,filename):
    """
    """
    import cv2
    import copy
    
    image_cv = copy.deepcopy(image)
    
    image_cv = image_cv.transpose(1, 2, 0)
    image_cv[:,:,0] += 103.939
    image_cv[:,:,1] += 116.779
    image_cv[:,:,2] += 123.68
    
    #print(np.amax(image_cv),np.amin(image_cv))

    
    cv2.imwrite(filename, image_cv)

    # from matplotlib import pyplot
    # import matplotlib as mpl
    # fig = pyplot.figure()
    # ax = fig.add_subplot(1,1,1)
    # # image = image.reshape(3,32,32).transpose(1,2,0)
    # imgplot = ax.imshow(image.T, cmap=mpl.cm.Greys)
    # imgplot.set_interpolation('nearest')
    # ax.xaxis.set_ticks_position('top')
    # ax.yaxis.set_ticks_position('left')
    # pyplot.savefig(filename)


def show(image):
    """
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    #image = image.reshape(3,32,32).transpose(1,2,0)
    imgplot = ax.imshow(image.T, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
    
