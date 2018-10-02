#!/bin/python
# python2.7
# Train and evaluate (audio-visual) model for the prediction of arousal / valence / liking
# The complexity of the SVM regressor is optimised.
# The performance on the development set in terms of CCC, PCE, and MSE is appended to corresponding text files (results.txt, results_pcc.txt, results_mse.txt).
# The predicitions on the test set are written into the folder specified by the variable 'path_test_predictions'.
# 
# Arguments (all optional):
#  argv[1]: Delay (in seconds, shift features to the back to compensate delay of annotations)
#  argv[2] argv[3] argv[4]: Each of these arguments is either 0 or 1 and determines if audio, video, and/or text features are taken into account
#  Default is: 0.0 1 1 1 
# 
# Contact: maximilian.schmitt@uni-passau.de

import os
import fnmatch
import numpy as np

from sys     import argv
from sklearn import svm

from load_features     import load_all
from calc_scores       import calc_scores
from write_predictions import write_predictions

# Set folders here
path_test_predictions = "D:/AVEC_17_Emotion_Sub-Challenge/result/"
b_test_available      = False  # If the test labels are not available, the predictions on test are written into the folder 'path_test_predictions'

# Folders wi o_features = "E:/AVEC_17_Emotion_Sub-Challenge/audio_features_xbow_6s/"
path_audio_features = "D:/AVEC_17_Emotion_Sub-Challenge/audio_features_xbow_6s/"
path_video_features = "D:/AVEC_17_Emotion_Sub-Challenge/audio_features_xbow_6s/"
path_text_features  = "D:/AVEC_17_Emotion_Sub-Challenge/text_features_xbow_6s/"
path_labels         = "D:/AVEC_17_Emotion_Sub-Challenge/labels/"

sr_labels = 0.1

delay = 0.0
b_audio = True
b_video = True
b_text  = True


print("Inside run_baseline.py")
print(len(argv))
if len(argv)>1:
     delay = float(argv[1])
if len(argv)>2:    
    b_audio = bool(int(argv[2]))
    b_video = bool(int(argv[3]))
    b_text  = bool(int(argv[4]))

print("argv")
print(argv)
print("delay")
print(delay)

path_features = []
if b_audio:
    path_features.append( path_audio_features )
if b_video:
    path_features.append( path_video_features )
if b_text:
    path_features.append( path_text_features )
        

if not b_test_available and not os.path.exists(path_test_predictions):
    os.mkdir(path_test_predictions)

# Compensate the delay (quick solution)
shift = int(np.round(delay/sr_labels))
shift = np.ones(len(path_features),dtype=int)*shift



files_train = fnmatch.filter(os.listdir(path_features[0]), "Train*")  # Filenames are the same for audio, video, text & labels
files_devel = fnmatch.filter(os.listdir(path_features[0]), "Devel*")
files_test  = fnmatch.filter(os.listdir(path_features[0]), "Test*")



print("i am going")
print(files_train)

# Load features and labels
Train   = load_all( files_train, path_features, shift )
Devel   = load_all( files_devel, path_features, shift )
Train_L = load_all( files_train, [ path_labels ] )  # Labels are not shifted
Devel_L = load_all( files_devel, [ path_labels ] )
#
#if b_test_available:
#    Test   = load_all( files_test, path_features, shift )
#    Test_L = load_all( files_test, [ path_labels ] )  # Test labels are not available in the challenge
#else:
#    Test   = load_all( files_test, path_features, shift, separate=True )  # Load test features separately to store the predictions in separate files
#
#
## Run liblinear (scikit-learn)
## Optimize complexity
num_steps = 16
complexities = np.logspace(-15,0,num_steps,base=2.0)  # 2^-15, 2^-14, ... 2^0
#
scores_devel_A = np.empty((num_steps,3))
scores_devel_V = np.empty((num_steps,3))
scores_devel_L = np.empty((num_steps,3))
#
seed = 0
#
for comp in range(0,num_steps):
    regA = svm.LinearSVR(C=complexities[comp],random_state=seed)
    regA.fit(Train,Train_L[:,0])
    predA = regA.predict(Devel)
    scores_devel_A[comp,:] = calc_scores(Devel_L[:,0],predA)
    
    regV = svm.LinearSVR(C=complexities[comp],random_state=seed)
    regV.fit(Train,Train_L[:,1])
    predV = regV.predict(Devel)
    scores_devel_V[comp,:] = calc_scores(Devel_L[:,1],predV)
#    
    regL = svm.LinearSVR(C=complexities[comp],random_state=seed)
    regL.fit(Train,Train_L[:,2])
    predL = regL.predict(Devel)
    scores_devel_L[comp,:] = calc_scores(Devel_L[:,2],predL)
#    
print("still going")    
#
ind_opt_A = np.argmax(scores_devel_A[:,0])
ind_opt_V = np.argmax(scores_devel_V[:,0])
ind_opt_L = np.argmax(scores_devel_L[:,0])
comp_opt_A = complexities[ind_opt_A]
comp_opt_V = complexities[ind_opt_V]
comp_opt_L = complexities[ind_opt_L]
#
## Run on train+devel with optimum complexity and predict on the test set
#TrainDevel   = np.concatenate((Train, Devel), axis=0)
#TrainDevel_L = np.concatenate((Train_L, Devel_L), axis=0)
#
#regA = svm.LinearSVR(C=comp_opt_A,random_state=seed)
#regA.fit(TrainDevel,TrainDevel_L[:,0])
#regV = svm.LinearSVR(C=comp_opt_V,random_state=seed)
#regV.fit(TrainDevel,TrainDevel_L[:,1])
#regL = svm.LinearSVR(C=comp_opt_L,random_state=seed)
#regL.fit(TrainDevel,TrainDevel_L[:,2])
#
#if b_test_available:
#    predA = regA.predict(Test)
#    predV = regV.predict(Test)
#    predL = regL.predict(Test)
#    score_test_A = calc_scores(Test_L[:,0],predA)
#    score_test_V = calc_scores(Test_L[:,1],predV)
#    score_test_L = calc_scores(Test_L[:,2],predL)
#else:
#    for f in range(0,len(files_test)):
#        predA = regA.predict(Test[f])
#        predV = regV.predict(Test[f])
#        predL = regL.predict(Test[f])
#        predictions = np.array([predA,predV,predL])
#        write_predictions(path_test_predictions,files_test[f],predictions,sr_labels)
#
#
## Print scores (CCC, PCC, RMSE) on the development set
print("Arousal devel (CCC,PCC,RMSE):")
print(scores_devel_A[ind_opt_A,:])
print("Valence devel (CCC,PCC,RMSE):")
print(scores_devel_V[ind_opt_V,:])
print("Liking  devel (CCC,PCC,RMSE):")
print(scores_devel_L[ind_opt_L,:])
#
#if b_test_available:
#    result_ccc  = [ scores_devel_A[ind_opt_A,0], score_test_A[0], scores_devel_V[ind_opt_V,0], score_test_V[0], scores_devel_L[ind_opt_L,0], score_test_L[0] ]
#    result_pcc  = [ scores_devel_A[ind_opt_A,1], score_test_A[1], scores_devel_V[ind_opt_V,1], score_test_V[1], scores_devel_L[ind_opt_L,1], score_test_L[1] ]
#    result_rmse = [ scores_devel_A[ind_opt_A,2], score_test_A[2], scores_devel_V[ind_opt_V,2], score_test_V[2], scores_devel_L[ind_opt_L,2], score_test_L[2] ]
#    print("Arousal test (CCC,PCC,RMSE):")
#    print(score_test_A)
#    print("Valence test (CCC,PCC,RMSE):")
#    print(score_test_V)
#    print("Liking  test (CCC,PCC,RMSE):")
#    print(score_test_L)
#else:
#    # Write only the scores for the development set
result_ccc  = [ scores_devel_A[ind_opt_A,0], scores_devel_V[ind_opt_V,0], scores_devel_L[ind_opt_L,0] ]
result_pcc  = [ scores_devel_A[ind_opt_A,1], scores_devel_V[ind_opt_V,1], scores_devel_L[ind_opt_L,1] ]
result_rmse = [ scores_devel_A[ind_opt_A,2], scores_devel_V[ind_opt_V,2], scores_devel_L[ind_opt_L,2] ]
#
## Write scores into text files
with open("results_ccc.txt", 'a') as myfile:
    #myfile.write(" Arousal Valence Liking\n")
    myfile.write(str(result_ccc) + '\n')
with open("results_pcc.txt", 'a') as myfile:
    #myfile.write("Arousal Valence Liking\n")
    myfile.write(str(result_pcc) + '\n')
#with open("results_rmse.txt", 'a') as myfile:
    #myfile.write("Arousal Valence Liking\n")
    #myfile.write(str(result_rmse) + '\n')
