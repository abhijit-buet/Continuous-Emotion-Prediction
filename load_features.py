#!/bin/python
# python2.7
# Contact: maximilian.schmitt@uni-passau.de

import numpy as np


def load_features( filenames, path, num_lines ):
    features  = np.empty((num_lines,get_num_columns(path + filenames[0]) - 2), float)
    
    c = 0
    for filename in filenames:
        with open(path + filename, 'r') as infile:
            for line in infile:
                pos = line.find(';', line.find(';') + 1)  # find second ; in order to remove instance name and time stamp
                features[c,:] = np.fromstring(line[pos+1:], dtype=float, sep=';')
                c += 1
    
    return features


def load_all( filenames, paths, shift=[], separate=False ):
    # Load the array filenames of CSV files for each folder in paths and concatenate them on axis 1
    # Features can be shifted on file level by the number of feature vectors specified in array shift
    # Shift rows to the future, pad first features with the first feature, cut the end - used for features in development and test partition
    # If the argument separate is True, the features are loaded into separate files.
    
    if len(shift)==0:
        shift = np.zeros(len(paths),dtype=int)
    
    if separate:
        F = np.empty((len(filenames)),dtype=np.object)
        for seq in range(0,len(filenames)):
            num_lines = get_num_lines(paths[0] + filenames[seq])
            F[seq] = load_features_shift( [filenames[seq]], paths[0], [num_lines], shift[0] )
            for p in range(1,len(paths)):
                Fn = load_features_shift( [filenames[seq]], paths[p], [num_lines], shift[p] )
                F[seq] = np.concatenate((F[seq],Fn), axis=1)
    else:  # Concatenate the features of all sequences into one array
        num_lines = get_num_lines_array(filenames, paths[0])
        F = load_features_shift( filenames, paths[0], num_lines, shift[0] )
        for p in range(1,len(paths)):
            Fn = load_features_shift( filenames, paths[p], num_lines, shift[p] )
            F = np.concatenate((F,Fn), axis=1)
    
    return F


def load_features_shift( filenames, path, num_lines, shift ):
    num_lines_sum = np.sum(num_lines)
    num_cols      = get_num_columns(path + filenames[0]) - 2
    
    features = np.empty((num_lines_sum,num_cols), float)
    
    f = 0
    c = 0
    for filename in filenames:
        c_start = c
        with open(path + filename, 'r') as infile:
            for line in infile:
                pos = line.find(';', line.find(';') + 1)  # find second ; in order to remove instance name and time stamp
                if c==c_start and shift>0:
                    fv = np.fromstring(line[pos+1:], dtype=float, sep=';')
                    for k in range(0,shift):
                        features[c,:] = fv
                        c += 1
                elif (c-c_start)==num_lines[f]:
                    break
                else:
                    features[c,:] = np.fromstring(line[pos+1:], dtype=float, sep=';')
                    c += 1
        f += 1
    
    return features


# Helper functions
def get_num_lines_array(filenames, path):
    num_lines = np.zeros(len(filenames),dtype=int)
    m = 0
    for filename in filenames:
        num_lines[m] = get_num_lines(path + filename)
        m += 1
    return num_lines

def get_num_lines(filename):
    with open(filename, 'r') as file:
        num_lines = 0
        for line in file:
            num_lines += 1
    return num_lines

def get_num_columns(filename,delim=';'):
    with open(filename, 'r') as file:
        line = file.readline()
    num_cols = line.count(delim) + 1
    return num_cols
