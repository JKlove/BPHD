import torch
import numpy
import matplotlib.pyplot as pt
#we can run the postProcessing for 1 Patient at the time. The ID doesn't always match the one of TABLE1 since the patients have been randomized.
Patient = 5
#Number of the GPU used. The script runs by exploiting cuda resources. This macro set the index of the GPU in your machine that will run the code.
torch.cuda.set_device(1)
#Number of Seizures for each patient. Column 3 of Table 1.
seizureN = [5, 4, 14, 4, 6, 2, 7, 3, 7, 13, 2, 10, 2, 10, 9, 2]
# K: column 3 of TABLE 3
crossValidationK = [4, 4, 12, 4, 6, 2, 5, 3, 4, 11, 2, 5, 2, 7, 8, 2]
# patient specific threshold described in paragraph 4.3
threshold = [-9.5/10,-9.5/10,-9.5/10,-9.5/10,-9.5/10,-8.5/10,-9.5/10,-7.5/10,\
			-9.5/10,-9.5/10,-9.5/10,-9.5/10,-8.5/10,-9.5/10,-9.5/10,-8.5/10]
# initilializations
number_seizures = range(seizureN[Patient-1])
cross = range(crossValidationK[Patient-1])
prediction = torch.zeros(1,1).cuda()
prediction = prediction[0,:]
counter1 = 0
l = 0
delay = 0
for j in cross:
	for i in number_seizures:
		#loading each file created with the test of a single seizure using each other seizure to train
		# Example with 2 seizure: we will have for example 2 different files for seizure 1, one trained with seizure 1, one with seizure 2
		# parameter i+1 points to the number of the seizure, parameter j to the number of the fold
		with open('/home/hyb/Desktop/HDC/HD/A/prediction/Pat' +str(Patient)+'/' + \
			'HDClassifier_Seizure'+str(i+1)+'Cross'+str(j)+'.txt', 'rb') as f1: #+'9K'
			prediction = torch.load(f1)
		Pred_mean = torch.cuda.FloatTensor(prediction.size()[0]).zero_()
		#Threhsolding the results inside the big 5 second window. Third step of Fig.2 of the paper. The Postprocessing rectangle.
		for i in numpy.arange(10,prediction.size()[0]):
		    Pred_mean[i] = torch.mean(prediction[i-10:i])
		# threshold that you find as last operation in the diagram (Fig.2)
		Pred_mean = torch.add(Pred_mean, threshold[Patient-1])
		Final_prediction = torch.sign(Pred_mean)
		# a simple method to compute the delay of detection: we subract from the first prevision the 180 seconds that precede the golden seizure onset.
		First = 1
		counter = 0
		for i in numpy.arange(0,Final_prediction.size()[0]):
			if Final_prediction[i] == 1 and First == 1 and i > 358:
				print(i/float(2))
				l = l+1
				delay = i/float(2)-180+delay
				First = 0
			# here we count the number of seizure prediction before the onset to compute the number of false alarms. The specificity is then computed as number of 1 - false alarms/360 with 360 number of interictal windows.
			# the mean specificity is then computed by only selecting the testing seizures (in the loop you cycle over all the possible seizures, also the trained ones. You have to select the correct one based on the fold in which you are)
			elif Final_prediction[i] == 1 and i < 360:
				counter = counter + 1
		counter1 = counter1+counter
		print('number of false:' + str(counter))
	print('end fold')
#total number of false detections in all the tests
print(counter1)
#raw delay estimation, considering also the seizures used for train
delay = delay/l
print('Mean delay: ' + str(delay))
