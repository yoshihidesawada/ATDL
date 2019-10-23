import os
import sys
import numpy as np

import macro as mc

# data normalization
def normalize(data):
	data = data/255.0
	return data

# shuffling keeping pair information
def paired_shuffle(data,labels,prop_num):

	zips = np.zeros((data.shape[0],data.shape[1]+labels.shape[1]+1), dtype=np.float32)
	for i in range(len(data)):
		zips[i] = np.hstack((labels[i],i,data[i]))
	np.random.shuffle(zips)

	slabels = zips[:,0:prop_num]
	sidx = zips[:,prop_num:prop_num+1]
	sdata = zips[:,prop_num+1:]

	return sdata, slabels, sidx
