#!/bin/python
# python2.7
# Write predictions of one sequence into csv files (delimiter ';')
# Arguments:
#  path_output: Path to store the output file, e.g., "test_predictions/"
#  filename:    Filename of the output file, e.g., "Test_01.csv"
#  predictions: Array of predictions (time x dimension), e.g., numpy.array([pred_arousal, pred_valence, pred_liking])
#  sr_Labels:   Sampling rate of the predictions in seconds (default: 0.1)
# 
# Contact: maximilian.schmitt@uni-passau.de

import os
import numpy as np

def write_predictions(path_output,filename,predictions,sr_labels=0.1):
    instancename = os.path.splitext(filename)[0]
    with open(path_output + filename, 'w') as file:
        for m in range(0,predictions.shape[1]):
            timestamp = m * sr_labels
            file.write("'" + instancename + "';" + str("%f" % timestamp))
            for n in range(0,predictions.shape[0]):
                file.write(";" + str("%f" % predictions[n,m]))
            file.write("\n")
