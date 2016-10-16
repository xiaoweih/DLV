#!/usr/bin/env python


import numpy as np

import h5py
import math
import display
import transform_NN as tnn
import solve
import basics

import sampling_basics as sb


boxX = sb.boxX
boxY = sb.boxY
populationSize = sb.populationSize

TWO_PI = 2 * np.pi

def analyze(model,x,y,xbounds,ybounds,plt):

    f = open('workfile', 'w')
    
    # the follwing is used to manually check the counterexamples
    #cex = [[4.519511723208989, 2.1549185960789887], [5.492689331075, 5.228611129355], [5.410794052446805, 5.04702293663], [1.41208851086, 4.38105551526], [3.203406112072587, 2.8692884599925867], [5.3428114013792625, 4.7272413875066235], [5.467069880662265, 5.25640636369], [5.439813752785, 5.036181697695], [3.2036080269943987, 2.8862026899843984], [5.735701836408878, 6.073247018308878], [5.713897558032118, 6.088903051442118], [5.487534611725, 5.252265063435], [3.2035779100550856, 2.883679821275086], [5.351601748933572, 4.7177042982735715], [5.404247649373935, 5.024659178213936], [5.720121169155047, 6.052874548035047], [5.352853450971872, 4.716346263911595], [5.348837409903864, 4.720703468423994], [5.411599283719501, 5.053884520355], [5.351601748933572, 4.7177042982735715], [4.12790833921, 1.77682779501], [4.2285734362229075, 1.7710596183729077], [4.126933675438968, 1.7873602620889675], [0.4700954109846791, 2.391321749615603], [4.128478698185452, 1.8065380356754521]]

    #for c in cex: basics.checkCex(model,c)
    
    initialGenX = np.random.random((populationSize, 1)) * TWO_PI
    initialGenY = np.random.random((populationSize, 1)) * TWO_PI
    initialGen = np.array([[initialGenX[k][0],initialGenY[k][0]] for k in range(populationSize) ])
    y_predicted = model.predict(initialGen)

    z = zip(initialGen,y_predicted)
    # 1 for the 1st output, and means that higher than the solver output
    # 2 for the 2nd output, and means that lower than the solver output
    xx = [ x for (x,y) in z if basics.chooseResult(y) == 1 ]
    xy = [ x for (x,y) in z if basics.chooseResult(y) == 2 ]

    xboxes = [ ([x1-boxX,y1-boxY],[x1+boxX,y1+boxY]) for [x1,y1] in xx]
    yboxes = [ ([x1-boxX,y1-boxY],[x1+boxX,y1+boxY]) for [x1,y1] in xy]

    (newxboxes,newyboxes,n) = sb.reproduce(model,plt,xboxes,yboxes,0)
    
    newxboxes = removeRedundancy(newxboxes)
    newyboxes = removeRedundancy(newyboxes)


    allcex = []

    cbs = sb.closestBoxes(newxboxes,newyboxes)
    for tb in cbs: 
       box = constructDangerousBox(tb)
       (result,cex) = solve.solve(model,f,box,2)
       if result == True: 
           #print "found counterexample!"
           cexInput = [ b for [x,b] in cex ]
           #print cexInput
           if basics.mapping(cexInput)[0] == True: 
               #print "input " + str(cexInput) + " is classified by the network as the 2nd output" 
               #print " but the actual value should be the 1st output"
               allcex.append(cexInput)

               
    cbs = sb.closestBoxes(newyboxes,newxboxes)
    for tb in cbs: 
       box = constructDangerousBox(tb)
       (result,cex) = solve.solve(model,f,box,1)
       if result == True: 
           #print "found counterexample!"
           cexInput = [ b for [x,b] in cex ]
           #print cexInput

           if basics.mapping(cexInput)[0] == False: 
               #print "input " + str(cexInput) + " is classified by the network as the 1st output" 
              # print " but the actual value should be the 2nd output"
               allcex.append(cexInput)

    if len(allcex) > 0: 
        cexX,cexY = zip(*allcex)
        plt.plot(cexX,cexY,'ro')
    else: 
        print "did not find counterexample in this round, try to check the stored model. "
               
    #basics.addPlotBoxes(plt,newxboxes,'r')
    #basics.addPlotBoxes(plt,newyboxes,'g')
    plt.savefig("pic/"+str(n)+".png", dpi=2400)
    
    print "The following set of points are the counterexamples between the neurual network and the actual function: "
    print allcex
    
    return model
    
    
def removeRedundancy(boxes): 
    remaining = []
    while len(boxes) > 0:
        box, boxes = boxes[0], boxes[1:] 
        remaining.append(box)
        for box2 in boxes: 
            if sb.pointWithinABox(sb.getMiddlePointOfABox(box2),box): 
                boxes.remove(box2)
    return remaining
         
def constructDangerousBox(twoBoxes):
    xbox = twoBoxes[0]
    ybox = twoBoxes[1] 

    xp = sb.getMiddlePointOfABox(xbox)
    yp = sb.getMiddlePointOfABox(ybox)
    
    if xp[0] >= yp[0] and xp[1] <= yp[1]: 
        return [xbox[0],ybox[0]]
    elif xp[0] <= yp[0] and xp[1] <= yp[1]:
        return [[xbox[1][0], ybox[0][1]],[ybox[1][0], xbox[0][1]]]
    elif xp[0] >= yp[0] and xp[1] >= yp[1]:
        return [[ybox[0][0], xbox[1][1]],[xbox[0][0], ybox[1][1]]]
    else: return [xbox[1],ybox[1]]