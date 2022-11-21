#!/usr/bin/env python

import sys
import numpy as np
import os
import scipy.io as scp
from iEEGFunctions import HD_classifier   #here there are all the functions that we use
import torch
import time
import pdb
import json
import matplotlib.pyplot as plt

#Number of the GPU used. The script runs by exploiting cuda resources. This macro set the index of the GPU in your machine that will run the code.
device = 1
torch.cuda.set_device(device)
#Patient 9, the third seizure is an artifact!
#number of seizures used to train the algorithm. Column 2 of Table 3
Seizure_train = [2, 1, 3, 1, 1, 1, 3, 1, 4, 3, 1, 6, 1, 4, 2, 1]
#directories with the dataset. You could change it based on the directory of your pc.
working_dir = '/home/data/MATLAB/'
#Number of Seizures for each patient. Column 3 of Table 1.
seizureN = [5, 4, 14, 4, 6, 2, 7, 3, 7, 13, 2, 10, 2, 10, 9, 2]
#frequencies at which the dataset is recorded. Refer to section 3.2 of the paper.
fs = 512
# K: column 3 of TABLE 3
crossValidationK = [4, 4, 12, 4, 6, 2, 5, 3, 4, 11, 2, 5, 2, 7, 8, 2]
#table that represents all the ictal segments used to train with each different
#patient and seizure. These timing have been choosen by visual inspection of the signal, trainng with the most suitable part of the seizure.
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
#dimension of the LBP = 6 (Section 4.1). T = dimLBP+1
T = 7	#1 + dimension l of binary pattern
totalNumberBP = 2**(T-1)
D = 10000   # dimension of hypervector
#List of patient that we want to test. You can add as many patient as you want
PatientList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
for Patient in PatientList:
	#loading first seizure from the target patient. The only important information is the nSignals, hence the number of channels per each patient that we use
	# to create the item memories. line 74-80 are hence used only to create the 2 item memories
	loading_file = working_dir + 'Pat'+str(Patient)+'/'+'Sz' +str(1)+'.mat'
	f = scp.loadmat(loading_file)
	EEGSez = torch.from_numpy(np.array(f['EEG'])).cuda().t()
	nSignals = EEGSez.size(0)
	torch.manual_seed(1);
	#creating the model with the 2 itemMemory, both random and binarized. Creation of C_1...C_64 e E_1...E_n descrived in section 4.2
	model = HD_classifier(totalNumberBP,D,nSignals,T,device, 'random')
	# one iteration for each fold inside the k-fold validation. It comprises one step of training with Seizure_train[patient] number of trained seizure
	# and one phase of testing with the remaining seizures tested.
	for cross in range(crossValidationK[Patient-1]):
			####TRAINING####
		for i in range(Seizure_train[Patient-1]):
			## loading correct seizure for training. In section 5.2 we describe the order. We first select Seizure_train[patient] number of trained seizure as the first seizure happening.
			## Then, during cross validation we rotate the trained seizure among all the available one in the temporal order (e.g. 7 seizures total, 2 seizures used for training. k = 6.
			# In the first fold seizures 1-2 training, seizures 3-7 test. Second fold 2-3 training seizures, 1+4-7 testing seizures and so on till 6-7 trained seizures and 1-5 tested one).
			loading_file = working_dir + 'Pat'+str(Patient)+'/'+'Sz' +str(i+1+cross)+'.mat'
			f = scp.loadmat(loading_file)
			#putting the EEG in torch on the cuda memory
			EEGSez = torch.from_numpy(np.array(f['EEG'][Seizure_begin[Patient-1][i]*fs:Seizure_end[Patient-1][i]*fs, :])).cuda().t()
			print ('Learning Seizure number ' + str(i+1+cross))
			#creating the prototype for ictal from the first seizure
			temp1=model.learn_HD_proj_big_half(EEGSez,fs)
			#I use the first seizure to create the HV: if we have more than 1, I will sum the other
			# to the first HV and perform an other binarization of the HV.
			if i == 0:
				queeryVectorS0 = temp1
			else:
				queeryVectorS0 = torch.add(queeryVectorS0,temp1)
		# I sum and threshold all the intermediate prototype to create the final one stored in the associative memory (green+blu part of the Fig.2 --> Associative Memory (AM), Ictal)
		queeryVectorS0 = (queeryVectorS0>Seizure_train[Patient-1]/2).short()
		#line 105-110: same procedure as above for the interictal prototype. Hence, creation of green+blu part of the Fig.2 --> Associative Memory (AM), Interictal
		loading_file = working_dir + 'Pat'+str(Patient)+'/'+'Sz1'+'.mat'
		f = scp.loadmat(loading_file)
		EEGInterictal = torch.from_numpy(np.array(f['EEG'][10*fs:50*fs,:])).cuda().t()
		print('Learning Interictal period')
		temp3=model.learn_HD_proj_big_half(EEGInterictal,fs)
		queeryVectornS0 = temp3
		    ####TESTING####
		i = 0
		j = 0
		t = time.time()
		## I rotate the test seizures among all the possible ones. Hence, I'm testing also the trained ones. During the postprocessing you have to select only the correct ones.
		for i in range(seizureN[Patient-1]):
			#loading of the recording and creation of the empty vectors with distances from ictal and interictal prototypes (last part of central box of Fig.2)
			loading_file = working_dir + 'Pat'+str(Patient)+'/'+'Sz' +str(i+1)+'.mat'
			f = scp.loadmat(loading_file)
			distanceVectorsS0 = torch.zeros(1,f['EEG'].shape[0]/fs*2).cuda()
			distanceVectorsnS0 = torch.zeros(1,f['EEG'].shape[0]/fs*2).cuda()
			#creation of the vector that goes out from the central box  of Fig.2 (HD computing: Encoding and Associative Memory)
			prediction0 = torch.zeros(1,f['EEG'].shape[0]/fs*2).cuda()
			#testing EEG
			EEGtest=torch.from_numpy(np.array(f['EEG'])).cuda().t()
			index = np.arange(0,EEGtest.size(1),fs/2)
			i = 0
			j = j+1
			#processing of each 0.5 second window inside the recording as explained in fig.2
			for iStep in index:
				#with this function we create the vector H that is then passed to the test phase (blue part of Fig.2) to compare with Hamming distance
				temp = model.learn_HD_proj(EEGtest[:,iStep:iStep+fs/2])
				#A simple comparison with hamming distance
				[distanceVectorsS0[0,i],distanceVectorsnS0[0,i],prediction0[0,i]] = model.predict(temp, queeryVectorS0, queeryVectornS0, D)
				i = i + 1
				print ('Patient:' +str(Patient)+ 'K-fold:' + str(cross+1)+'Seizure: ' + str(j) +'; half second: ' + str(i))
			#for the purpose of this code, we save only prediction, discarding the 2 distances from the 2 prototypes
			#These will be used in next draft of the project.
			with open('/home/hyb/Desktop/HDC/HD/A/LBHD_prediction/Pat'+\
				str(Patient)+'/' + 'HDClassifier_Seizure'+str(j)+'Cross'+str(cross)+'.txt', 'wb') as f1:
				torch.save(prediction0[0,:],f1)
##All the directories refer to the ETHZ network: of course you have to change in the appropriate way.
