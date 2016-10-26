# DLV


Note: the software is currently under active development. Please feel free to contact the developer by email: xiaowei.huang@cs.ox.ac.uk. 

Together with the software, there are two documents, one is a paper in arXiv and the other is an user document. The user document will be updated from time to time. Please refer to the documents for more details about the software. 

(1) Installation: 

To run the program, one needs to install the following packages: 

Python 2.7
Theano
Keras

Moreover, to run experiments on ImageNet, one needs to download the VGG16 pre-trained model from 

https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

and download images and put into the directory networks/imageNet/, and change the corresponding paths in networks/imageNet_network.py

(2) Usage: 

Use the following command to call the program: 

           python main.py

Please use the file ''configuration.py'' to set the parameters for the system to run. 


Xiaowei Huang