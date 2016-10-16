#!/usr/bin/env python



import numpy as np

def writeFile(model,f):

    for layer in model.layers:
     g=layer.get_config()
     h=layer.get_weights()
     ws = ''
     for w in h: 
         ws += np.array2string(w,formatter={'float_kind':lambda x: "%.6f" % x}) + "\n"
     f.write ("config=\n"+str(g)+"\n")
     f.write ("weight=\n"+ws+"\n")
     l = layer.get_input_at(0)
     l = layer.get_input_shape_at(0)
     f.write ("input at 0=\n"+str(l)+"\n")
     f.write ("input shape at 0=\n"+str(l)+"\n")
     
     f.write ("\nHmmm, moving to another layer ... "+"\n\n")




def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape))
    finally:
        f.close()
        
        
def printVector(vs): 

    for i in vs: 
        print i
         