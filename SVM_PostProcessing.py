import torch
import numpy
import matplotlib.pyplot as pt
#we can run the postProcessing for 1 Patient at the time.
Patient = 16
#Number of the GPU used
torch.cuda.set_device(1)
#Number of Seizures for each patient
seizureN = [5, 4, 14, 4, 6, 2, 7, 3, 7, 13, 2, 10, 2, 10, 9, 2]
#crossValidationK = [4, 4, 1, 4, 6, 2, 1, 3, 1, 1, 2, 1, 2, 1, 8, 2]
crossValidationK = [4, 4, 12, 4, 6, 2, 5, 3, 4, 11, 2, 5, 2, 7, 8, 2]
threshold = [-9.5/10,-9.5/10,-9.5/10,-9.5/10,-9.5/10,-8.5/10,-9.5/10,-7.5/10,\
			-9.5/10,-9.5/10,-9.5/10,-9.5/10,-8.5/10,-9.5/10,-9.5/10,-8.5/10]
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
		with open('/home/hyb/Desktop/HDC/HD/A/SVM_prediction/Pat' +str(Patient)+'/' + \
			'SVMClassifier_Seizure'+str(i+1)+'Cross'+str(j)+'.txt', 'rb') as f1: #+'9K'
			prediction = torch.load(f1)
		Pred_mean = numpy.array(torch.cuda.FloatTensor(prediction.size).zero_().cpu().numpy())
		#Threhsolding the results inside the big 5 second window
		for i in numpy.arange(10,prediction.size):
		    Pred_mean[i] = numpy.mean(prediction[i-10:i])
		Pred_mean = Pred_mean+ threshold[Patient-1]
		Final_prediction = numpy.sign(Pred_mean)
		First = 1
		counter = 0
		for i in numpy.arange(0,Final_prediction.size):
			if Final_prediction[i] == 1 and First == 1 and i > 358:
				print(i/float(2))
				l = l+1
				delay = i/float(2)-180+delay
				First = 0
			elif Final_prediction[i] == 1 and i < 350:
				counter = counter + 1
		counter1 = counter1+counter
		print('number of false:' + str(counter))
	print ('end fold' + str(j+1))
#total number of false detections in all the tests
print (counter1)
#raw delay estimation, considering also the seizures used for train
delay = delay/l
print ('Mean delay: ' + str(delay))