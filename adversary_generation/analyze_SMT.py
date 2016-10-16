#!/usr/bin/env python


import numpy as np

import h5py

import display
import transform_NN as tnn
import solve
import basics

step = 2
samplesPerBox = 100
minX = 1
minY = 1

def analyze(model0,x,y,xbounds,ybounds,plt):

    model = tnn.transform(model0)

    f = open('workfile', 'w')

    f.write ("Original Model:\n\n")
    display.writeFile(model0,f)
    
    f.write ("New Model:\n\n")
    display.writeFile(model,f)
    
    boxesToReport = []
    boxesForFurtherProcess = []
    
    mapping = basics.mapping
    
    boxes = recursiveBoxVerification(model0,f,mapping,boxesToReport,boxesForFurtherProcess,xbounds,ybounds,step)
    
    basics.addPlotBoxes(plt,boxes,'r')
    
    f.close()
     
    return model
    
def recursiveBoxVerification(model0,f,mapping,boxesToReport,boxesForFurtherProcess,xbounds,ybounds,step):    
    print str(len(boxesToReport)) + " bad boxes have been found so far ... "
    print "boxes remaining to be processed: " + str(len(boxesForFurtherProcess))
    print "the size of the current box: " + str(xbounds[1]-xbounds[0])

    boxes = generateBoxes(xbounds,ybounds,step)
    (badBoxes,goodBoxes) = verifyBoxes(model0,f,boxes)  
    badBoxes = [ bb for bb in badBoxes if compareBoxSize(bb) ]
    boxesForFurtherProcess = boxesForFurtherProcess + badBoxes      
    for bb in goodBoxes: 
        if validateBox(model0,f,mapping,bb): 
           boxesToReport.append(bb)
        elif compareBoxSize(bb): 
           print "adding box ... " + str(bb)
           boxesForFurtherProcess.append(bb)
    if len(boxesForFurtherProcess) > 0 :
        newbox, boxesForFurtherProcess = boxesForFurtherProcess[0], boxesForFurtherProcess[1:]
        newxbound = [newbox[0][0],newbox[1][0]]
        newybound = [newbox[0][1],newbox[1][1]]
        return recursiveBoxVerification(model0,f,mapping,boxesToReport,boxesForFurtherProcess,newxbound,newybound,step)
    else: return boxesToReport

# the size is big enough for a further processing
def compareBoxSize(bb): 
    return (minX < (bb[1][0] - bb[0][0])) or (minY < (bb[1][1] - bb[0][1]))
    
def generateBoxes(xbounds,ybounds,n):
    xgap = (xbounds[1]-xbounds[0])/n
    xs = [xbounds[0] + xgap*k for k in range(n)] 
    ygap = (ybounds[1]-ybounds[0])/n
    ys = [ybounds[0] + ygap*k for k in range(n)] 
    boxes = [[[x,y],[x+xgap,y+ygap]] for x in xs  for y in ys]
    return boxes
    
# differentiating boxes with and without the solver curve crossing 
def verifyBoxes(model0,f,boxes): 
    badboxes = []
    goodboxes = []
    i = 0
    for box in boxes: 
        #print "processing "+str(i+1)+"th box " + str(box) + " ... "
        [a0,m0] = solve.solve(model0,f,box,2)
        [a1,m1] = solve.solve(model0,f,box,1)
        if a0 and a1: 
            badboxes.append(box)
        else: goodboxes.append(box)
        i += 1
    #print str(len(badboxes)) + " boxes are bad, among all "+ str(len(boxes)) + " boxes."
    return (badboxes,goodboxes)

# decide whether a box is a definite bad box
def validateBox(model0,f,mapping,box):
    ps0 = []
    ps1 = []
    for i in range(samplesPerBox):
	    x = sampleAPointFromABox(box)
	    y = mapping(x)
	    y1 = basics.normalisation(y)
	    if y1[0] > y1[1]: 
	        ps0.append(x)
	    else:
	    	ps1.append(x)
    [a0,m0] = solve.solve(model0,f,box,2)
    [a1,m1] = solve.solve(model0,f,box,1)
    return (len(ps0) == 0 and a1 == True) or (len(ps0) == 1 and a0 == True)

	
	
def sampleAPointFromABox(box):
    xr = np.random.random_sample()
    yr = np.random.random_sample()
    x = box[0][0] + (box[1][0]-box[0][0]) * xr
    y = box[0][1] + (box[1][1]-box[0][1]) * yr
    return [x,y]

    

