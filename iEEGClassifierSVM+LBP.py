#!/usr/bin/env python

import sys
import numpy as np
import os
import scipy.io as scp
import iEEGFunctions  #here there are all the functions that we use
import torch
import time
import pdb
import json
import matplotlib.pyplot as plt
from sklearn import svm
#Number of the GPU used
device = 1
#Patient 9, the third seizure is an artifact!
Seizure_train = [2, 1, 3, 1, 1, 1, 3, 1, 4, 3, 1, 6, 1, 4, 2, 1]
torch.cuda.set_device(device)
working_dir = '/home/data/MATLAB/'
seizureN = [5, 4, 14, 4, 6, 2, 7, 3, 7, 13, 2, 10, 2, 10, 9, 2]
fs = 512
#table that represents all the ictal segments used to train with each different 
#patient and seizure
Seizure_begin = [[184, 184, 183, 186, 186, 0,   0,   0,   0,   0, 0, 0, 0, 0],\
					[180, 192, 188, 186, 0, 0,   0,   0,   0,   0, 0, 0, 0, 0],\
					[181, 207, 231, 190, 190, 196, 193, 202, 205, 201, 195, 195, 194, 191],\
					[220, 219, 200, 192, 0, 0,   0,   0,   0,   0, 0, 0, 0, 0],\
					[187, 188, 208, 202, 202, 212,   0,   0,   0,   0, 0, 0, 0, 0],\
					[184, 186, 0, 0, 0, 0,   0,   0,   0,   0, 0, 0, 0, 0],\
					[200, 210, 200, 200, 220, 200, 200,   0,   0,   0, 0, 0, 0, 0],\
					[185, 188, 185, 0, 0, 0,   0,   0,   0,   0, 0, 0, 0, 0],\
					[188, 204, 204, 214, 183, 203, 185,   0,   0,   0, 0, 0, 0, 0],\
					[181, 185, 182, 182, 181, 182,   180,   184, 181, 180, 181, 182, 180, 0],\
					[181, 180, 0, 0, 0, 0,   0,   0,   0,   0, 0, 0, 0, 0],\
					[210, 210, 200, 220, 200, 200, 210, 200, 200, 190, 0, 0, 0, 0],\
					[185, 185, 0, 0, 0, 0,   0,   0,   0,   0, 0, 0, 0, 0],\
					[195, 185, 194, 195, 184, 185, 235, 185, 182, 185, 0, 0, 0, 0],\
					[213, 208, 200, 204, 210, 200, 216, 205, 203,   0, 0, 0, 0, 0],\
					[205, 195, 0, 0, 0, 0,   0,   0,   0,   0, 0, 0, 0, 0]]
Seizure_end = [[192, 193, 189, 190, 200, 0,   0,   0,   0,   0, 0, 0, 0, 0],\
					[210, 215, 215, 215 , 0, 0,   0,   0,   0,   0, 0, 0, 0, 0],\
					[200, 230, 255, 220, 219, 219, 215, 228, 228, 226, 220, 213, 218, 210],\
					[250, 255, 239, 217, 0, 0,   0,   0,   0,   0, 0, 0, 0, 0],\
					[225, 218, 234, 231, 221, 242,   0,   0,   0,   0, 0, 0, 0, 0],\
					[193, 195, 0, 0, 0, 0,   0,   0,   0,   0, 0, 0, 0, 0],\
					[220, 230, 220, 220, 240, 225, 220,   0,   0,   0, 0, 0, 0, 0],\
					[205, 205, 210, 0, 0, 0,   0,   0,   0,   0, 0, 0, 0, 0],\
					[201, 231, 210, 244, 195, 226, 211,   0,   0,   0, 0, 0, 0, 0],\
					[189, 190, 196, 190, 183, 188,   181,   193, 190, 189, 191, 199, 181, 0],\
					[211, 210, 0, 0, 0, 0,   0,   0,   0,   0, 0, 0, 0, 0],\
					[225, 225, 215, 230, 220, 220, 220, 220, 220,   0, 0, 0, 0, 0],\
					[210, 210, 0, 0, 0, 0,   0,   0,   0,   0, 0, 0, 0, 0],\
					[210, 200, 218, 219, 214, 220, 268, 220, 222, 222, 0, 0, 0, 0],\
					[238, 225, 225, 234, 235, 225, 241, 225, 233,   0, 0, 0, 0, 0],\
					[235, 225, 0, 0, 0, 0,   0,   0,   0,   0, 0, 0, 0, 0]]					
second = fs
minutes = second*60
T = 7 #1 + dimension l of binary pattern
totalNumberBP = 2**(T-1)
#List of patient that we want to test
PatientList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
#crossValidationK = [4, 4, 1, 4, 6, 2, 1, 3, 1, 1, 2, 1, 2, 1, 8, 2]
crossValidationK = [4, 4, 12, 4, 6, 2, 5, 3, 4, 11, 2, 5, 2, 7, 8, 2]
for Patient in PatientList:
	loading_file = working_dir + 'Pat'+str(Patient)+'/'+'Sz' +str(1)+'.mat'
	f = scp.loadmat(loading_file)
	EEGSez = torch.from_numpy(np.array(f['EEG'])).cuda().t()
	nSignals = EEGSez.size(0)
	torch.manual_seed(1);
	for cross in range(crossValidationK[Patient-1]):
		for i in range(Seizure_train[Patient-1]):
			####TRAINING####
			loading_file = working_dir + 'Pat'+str(Patient)+'/'+'Sz' +str(i+1+cross)+'.mat'
			f = scp.loadmat(loading_file)
			EEGSez = torch.from_numpy(np.array(f['EEG'][Seizure_begin[Patient-1][i]*fs:Seizure_end[Patient-1][i]*fs, :])).cuda().t()
			print('Learning Seizure number ' + str(i+1+cross))
			temp1=iEEGFunctions.LBP_window_train(EEGSez,fs, totalNumberBP, 1,T)
			if i == 0:
				queeryVectorS0 = temp1	
			else:
				queeryVectorS0 = torch.cat((queeryVectorS0, temp1),0)
		loading_file = working_dir + 'Pat'+str(Patient)+'/'+'Sz1'+'.mat'
		f = scp.loadmat(loading_file)
		EEGInterictal = torch.from_numpy(np.array(f['EEG'][10*fs:50*fs,:])).cuda().t()
		print('Learning Interictal period')
		temp3=iEEGFunctions.LBP_window_train(EEGInterictal,fs, totalNumberBP, 0,T)
		queeryVectornS0 = temp3
		Matrix_train = torch.cat((queeryVectorS0, queeryVectornS0),0)
		#the column 0 contains the labels, the other columns contain the futures
		y = Matrix_train[:,0]
		X = Matrix_train[:,1:totalNumberBP*nSignals+1]
		#linear SVM classifier: changing c we will not impact the results so much
		clf = svm.SVC(kernel='linear', C = 1.0) #if results are different make c = 0.1 
		clf.fit(X.cpu().numpy(), y.cpu().numpy())
		    ####TESTING####		    
		i = 0
		j = 0
		t = time.time()
		for i in range(seizureN[Patient-1]):
			loading_file = working_dir + 'Pat'+str(Patient)+'/'+'Sz' +str(i+1)+'.mat'
			f = scp.loadmat(loading_file)
			distanceVectorsS0 = torch.zeros(1,f['EEG'].shape[0]/fs*2).cuda()
			distanceVectorsnS0 = torch.zeros(1,f['EEG'].shape[0]/fs*2).cuda()
			prediction0 = np.array(torch.zeros(f['EEG'].shape[0]/fs*2).cuda().cpu().numpy())
			EEGtest=torch.from_numpy(np.array(f['EEG'])).cuda().t()
			index = np.arange(0,EEGtest.size(1),fs/2)
			i = 0
			j = j+1
			for iStep in index:
				temp = iEEGFunctions.LBP_extractor(EEGtest[:,iStep:iStep+fs/2],totalNumberBP,2,T)
				#here the label is totally random, so we simply discard it: we want to predict the right one.
				X_test = temp[1:totalNumberBP*nSignals+1]
				prediction0[i] = clf.predict(X_test.t().cpu().numpy())
				i = i + 1
				print ('Patient:' +str(Patient)+ 'K-fold:' + str(cross+1)+'Seizure: ' + str(j) +'; half second: ' + str(i))
			with open('/home/hyb/Desktop/HDC/HD/A/SVM_prediction/Pat'+\
				str(Patient)+'/' + 'SVMClassifier_Seizure'+str(j)+'Cross'+str(cross)+'.txt', 'wb') as f1:
				torch.save(prediction0,f1)
##All the directories refer to the ETHZ network: of course you have to change in the appropriate way.
