#!/usr/bin/env python


import numpy as np

import math
import basics


samplesPerBox = 300
boxX = 0.03
boxY = 0.03
movePace = 0.06

populationSize = 1000

TWO_PI = 2 * np.pi

    
def reproduce(model,plt,xboxes,yboxes,n):

    print str(n)+"th reproduction ... "
    
    #basics.addPlotBoxes(plt,xboxes,'r')
    #basics.addPlotBoxes(plt,yboxes,'g')
    #plt.savefig("pic/"+str(n)+".png")

    dists = {}
    for box1 in xboxes: 
       for box2 in yboxes: 
           d = distanceBox(box1,box2)
           dists[str(box1),str(box2)] = d
           dists[str(box2),str(box1)] = d

    gaps = 0

    newxboxes = []
    for box in xboxes: 
        if checkBox(model,box) != -1: 
            (box2,d) = closestBox(box,yboxes,dists)
            if d != 0: (newbox,gap) = moveTowards(model,box,box2,d)
            else: (newbox,gap) = (box,0)
            newxboxes.append(newbox)
            gaps += gap
        else: newxboxes.append(box)
    
    newyboxes = []
    for box in yboxes: 
        if checkBox(model,box) != -1: 
            (box2,d) = closestBox(box,xboxes,dists)
            if d != 0: (newbox,gap) = moveTowards(model,box,box2,d)
            else: (newbox,gap) = (box,0)
            newyboxes.append(newbox)
            gaps += gap
        else: newyboxes.append(box)
    
    if gaps < 1: return (newxboxes,newyboxes,n+1)
    else: return reproduce(model,plt,newxboxes,newyboxes, n+1)


# whether all samples in a box return the same value
def checkBox(model,box):
    xs = np.random.random((samplesPerBox, 1)) * (box[1][0]-box[0][0])
    ys = np.random.random((samplesPerBox, 1)) * (box[1][1]-box[0][1])
    ps = np.array([[box[0][0] + xs[k][0],box[0][1] + xs[k][0]] for k in range(samplesPerBox) ])
    y_predicted = model.predict(ps) 
    
    xx = [ x for (x,y) in zip(ps,y_predicted) if basics.chooseResult(y) == True ]
    if len(xx) == 0: return 0
    elif len(xx) == samplesPerBox : return 1
    else: return -1

def distanceBox(box1,box2): 
    p1 = getMiddlePointOfABox(box1)
    p2 = getMiddlePointOfABox(box2)
    return distance(p1,p2)
    
def pointWithinABox(p,box): 
    p1 = getMiddlePointOfABox(box)
    d = distance(p,p1)
    return d < abs(p[0] - box[0][0]) and d < abs(p[1] - box[0][1])
    
def getMiddlePointOfABox(box): 
    x = box[0][0] + (box[1][0] - box[0][0]) / 2
    y = box[0][1] - (box[0][1] - box[1][1]) / 2
    return [x,y]
    
def fromMiddlePointGetBox(p): 
    return [[p[0]-boxX,p[1]+boxY],[p[0] + boxX,p[1] - boxY]]
    
def distance(p1,p2): 
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    
def closestBox(box,boxes,dists): 
    k = [box,2*TWO_PI**2]
    for box2 in boxes: 
        d = dists[str(box),str(box2)]
        if d < k[1]:
            k = [box2,d]
    return k
    
def closestBoxes(xboxes,yboxes): 
    bbs = []
    dists = {}
    for box1 in xboxes: 
       for box2 in yboxes: 
           d = distanceBox(box1,box2)
           dists[str(box1),str(box2)] = d
    for box1 in xboxes: 
        [box2, d] = closestBox(box1,yboxes,dists)
        bbs.append([box1,box2])
    return bbs

def moveTowards(model,box1,box2,d):
    pace = movePace / d
    mid1 = getMiddlePointOfABox(box1)
    mid2 = getMiddlePointOfABox(box2)
    xdistance = mid2[0] - mid1[0]
    ydistance = mid2[1] - mid1[1]
    mid = [mid1[0] + pace * xdistance, mid1[1] + pace * ydistance]
    newbox = fromMiddlePointGetBox(mid)
    if checkBox(model,box1) == checkBox(model,newbox) and checkBox(model,box1) != -1 : 
        return (newbox,distance(mid1,mid))
    else: return (box1,0)