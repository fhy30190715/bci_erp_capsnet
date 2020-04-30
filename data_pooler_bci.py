# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:03:43 2020

@author: Kassymzhomart Kunanbayev aka @qasymjomart

"""

import mne
import pickle
import numpy as np
import pandas
from train_utils_bci import subject_specific, pad_with_zeros_below, pad_by_duplicating, concatenate_array, concatenate_dicts #import functions written by hand
from sklearn.model_selection import train_test_split

#%%


#%%
def normalization(x_all, y_all, test_subjects, val_subjects):

	x_train,temp1,y_train,temp2 = train_test_split(x_all, y_all, test_size = 0.00000001, random_state = 42, shuffle = True)

	#mu = np.mean(train,axis=0)
	#stdev = np.std(train,axis=0)

	train_norm = np.random.rand(x_train.shape[0],x_train.shape[1],x_train.shape[2])
	a1 = np.random.rand(x_train.shape[1],x_train.shape[2])
	a2 = np.random.rand(x_train.shape[1],x_train.shape[2])

	#val_norm = np.random.rand(x_val.shape[0],x_val.shape[1],x_val.shape[2])
	#b1 = np.random.rand(x_val.shape[1],x_val.shape[2])
	#b2 = np.random.rand(x_val.shape[1],x_val.shape[2])



	#test_norm = np.random.rand(x_test.shape[0],x_test.shape[1],x_test.shape[2])
	#c1 = np.random.rand(x_test.shape[1],x_test.shape[2])
	#c2 = np.random.rand(x_test.shape[1],x_test.shape[2])

	for i in range(x_train.shape[1]):
		for j in range(x_train.shape[2]):
			a1[i,j] = np.min(x_train[:,i,j])
			a2[i,j] = np.max(x_train[:,i,j])
			#b1[i,j] = np.min(val[:,i,j])
			#b2[i,j] = np.max(val[:,i,j])
			#c1[i,j] = np.min(test[:,i,j])
			#c2[i,j] = np.max(test[:,i,j])

	def weird_division(n, d):
		return n / d if d else 0

	for i in range(x_train.shape[0]):
		for j in range(x_train.shape[1]):
			for k in range(x_train.shape[2]):
				train_norm[i,j,k] = weird_division((x_train[i,j,k] - a1[j,k]),(a2[j,k] - a1[j,k]))


	'''for i in range(x_val.shape[0]):
	for j in range(x_val.shape[1]):
	for k in range(x_val.shape[2]):
	val_norm[i,j,k] = weird_division((x_val[i,j,k] - a1[j,k]),(a2[j,k] - a1[j,k]))'''

	for x in range(len(val_subjects)):
		for i in range(val_subjects[x]['xtrain'].shape[0]):
			for j in range(val_subjects[x]['xtrain'].shape[1]):
				for k in range(val_subjects[x]['xtrain'].shape[2]):
					val_subjects[x]['xtrain'][i,j,k] = weird_division((val_subjects[x]['xtrain'][i,j,k] - a1[j,k]),(a2[j,k] - a1[j,k]))   

	"""for i in range(x_test.shape[0]):
	for j in range(x_test.shape[1]):
	for k in range(x_test.shape[2]):
	test_norm[i,j,k] = weird_division((x_test[i,j,k] - a1[j,k]),(a2[j,k] - a1[j,k]))"""

	for x in range(len(test_subjects)):
		for i in range(test_subjects[x]['xtrain'].shape[0]):
			for j in range(test_subjects[x]['xtrain'].shape[1]):
				for k in range(test_subjects[x]['xtrain'].shape[2]):
					test_subjects[x]['xtrain'][i,j,k] = weird_division((test_subjects[x]['xtrain'][i,j,k] - a1[j,k]),(a2[j,k] - a1[j,k]))



	return train_norm, y_train, val_subjects,test_subjects
