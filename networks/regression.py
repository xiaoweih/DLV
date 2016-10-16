#!/usr/bin/env python

"""
For Xiaowei to train MLP for regression
"""
import sys
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense
import keras.optimizers

# visualisation
from keras.utils.visualize_util import plot

#
#import analyze  as analyzeNN
#import display
import basics 
from math import sqrt


TWO_PI = 2 * np.pi

studyRangeLow = [0,0]
studyRangeHigh = [TWO_PI,TWO_PI]

def target_fun(x):
    # function to regress
    #p = 3*np.sin(10*x[0]) + 0.2*(x[0]**2) + 1
    p = 1 + sqrt(10*(abs(x[0]-3)))
    return p
 
def LABELS(index):
    labels = ['0', '1']
    return labels[index]
    
def save(layer,image,filename):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    plt.plot(np.linspace(studyRangeLow[0], studyRangeHigh[0]), [target_fun([x]) for x in np.linspace(studyRangeLow[0], studyRangeHigh[0])], 'r')
    if layer == -1: 
        color = 'g.'
    elif layer == 0: 
        color = 'r.'
    elif layer == 1:
        color = 'b.'
    elif layer == 2:
        color = 'y.'
    elif layer == 3:
        color = 'c.'
    else: color = 'b.'
    plt.plot([image[0]], [image[1]], color)
    plt.savefig(filename)


"""
def regression():

    if len(sys.argv) > 1: 
        if sys.argv[1] == '0': 
            fromFile = False
        else: fromFile = True
    else: fromFile = False
    
    N_samples = 5000
    N_tests = 1000

    # load data
    x_train, y_train, x_test, y_test = NN.load_data(N_samples,N_tests)
    
    if fromFile == False: 

        # define and construct model
        print "Building network model ......"
        model = NN.build_model()
    
        plot(model, to_file='model.png')

        # visualisation

        # configure learning process
        sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss={'output': 'mse'})

        model.summary()

        start_time = time.time()
        model.fit({'data': x_train}, {'output': y_train}, nb_epoch=3000, validation_split=0.1, verbose=0)
        print("Fitting time: --- %s seconds ---" % (time.time() - start_time))
    
        print("Training finished!")

        # save model
        json_string = model.to_json()
        open('MLP.json', 'w').write(json_string)
        model.save_weights('MLP.h5', overwrite=True)
        sio.savemat('MLP.mat', {'weights': model.get_weights()})
        print("Model saved!")

    else: 
        print("Start loading model ... ")
        model = basics.read_model_from_file('MLP.mat','MLP.json')
        model.summary()


    #display.print_structure("MLP.h5")
    print("Start analyzing model ... ")
    start_time = time.time()
    model1 = analyzeNN.analyze(model, x_train, y_train, [studyRangeLow[0],studyRangeHigh[0]], [studyRangeLow[1],studyRangeHigh[1]], plt)
    print("Analyzing time: --- %s seconds ---" % (time.time() - start_time))

    model1.save_weights('MLP1.h5', overwrite=True)
    #analyzeNN.print_structure("MLP1.h5")

    # prediction after training
    y_predicted = model.predict(x_test)
    y_predicted1 = model1.predict(x_test)

    # display results
    
    plt.plot(np.linspace(studyRangeLow[0], studyRangeHigh[0]), [basics.target_fun([x]) for x in np.linspace(studyRangeLow[0], studyRangeHigh[0])], 'r')
    train_set = zip(x_train,y_train)
    train_set_high = [ x for (x,y) in train_set if y[0] == 1]
    train_set_low = [ x for (x,y) in train_set if y[0] == 0]
    (x_train_high, y_train_high) = zip(*train_set_high)
    (x_train_low, y_train_low) = zip(*train_set_low)
    plt.plot(x_train_high, y_train_high, 'g.')
    plt.plot(x_train_low, y_train_low, 'y.')
    
    threshold = 0.5
    dangerThreshold = 0.9
    plt.savefig("pic/result.png")
    plt.show()
    
"""

"""
    test_set = zip(x_test,y_test,y_predicted)
    test_set_high = [ x for (x,y,z) in test_set if y[0] == True and z[0] >= threshold ]
    test_set_low = [ x for (x,y,z) in test_set if y[0] == False and z[0] <= 1 - threshold ]
    test_set_wrong = [ x for (x,y,z) in test_set if ~(y[0] == True and z[0] >= threshold) and ~(y[0] == False and z[0] <= 1 - threshold) ] 
    test_set_wrong2 = [ x for (x,y,z) in test_set if (y[0] == True and z[0] <= 1- dangerThreshold) or (y[0] == False and z[0] >= dangerThreshold) ] 


    print str(len(test_set_wrong)) + " testing samples are classified wrong, among all " + str(len(test_set)) +" samples "
    print str(len(test_set_wrong2)) + " testing samples are classified wrong in a definite way, among all " + str(len(test_set)) +" samples "

    if len(test_set_high) > 0: 
        (x_test_high, y_test_high) = zip(*test_set_high)
        plt.plot(x_test_high, y_test_high, 'c.')
    if len(test_set_low) > 0: 
        (x_test_low, y_test_low) = zip(*test_set_low)
        plt.plot(x_test_low, y_test_low, 'b.')

    (x_test_wrong, y_test_wrong) = zip(*test_set_wrong)
    plt.plot(x_test_wrong, y_test_wrong, 'r.')
"""
    #plt.legend(['Target line', 'Training samples high', 'Training samples low', 'Testing samples high', 'Testing samples low', 'Testing samples wrong' ])
    



