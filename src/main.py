import os
import sys

import random
import numpy as np
import pandas as pd

# Please modify to fit your environment
import tensorflow as tf
import tensorflow.contrib.keras.api.keras as keras
from tensorflow.contrib.keras.api.keras import backend, callbacks
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Input
from tensorflow.contrib.keras.api.keras.utils import Progbar
from tensorflow.contrib.keras.api.keras.optimizers import Adam

from functools import partial

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import macro as mc
import load_data as ld
import preprocess as pre
import models
import compute_relation_vectors as rv

if len(sys.argv) != 2:
	print("input error: main.py method_flag")
	print("method flag : nontransfer (=0), standard transfer learning (=1), count ver. all transfer deep learning (=2),\
	mean ver. all transfer deep learning (=3), mean modified ver. all transfer deep learning (=4)")
	sys.exit(1)
_, method_flag = sys.argv

def Neighbors( labels, database, knum ):
	nbrs = NearestNeighbors(n_neighbors=knum, algorithm='ball_tree').fit(database)
	dis, idx = nbrs.kneighbors(labels)
	return dis, idx

def main(method_flag):
	# load data
	source_df, target_df = ld.load_file()

	predicts, corrects = [], []

	random.seed(123)
	np.random.seed(123)

	kf = KFold(shuffle=False,random_state=1,n_splits=mc._FOLD_NUM)
	fold_num = 1
	cnt = 0
	for train, test in kf.split(target_df):
		print('{0}/{1}'.format(fold_num, mc._FOLD_NUM))
		target_train = target_df.iloc[train]
		target_test = target_df.iloc[test]

		idx, labels = transfer_model(source_df, target_train, target_test, method_flag, fold_num)
		predicts.extend(idx.tolist())
		corrects.extend(labels[0].tolist())
		fold_num = fold_num+1

	# save results
	predicts = np.array(predicts)
	corrects = np.array(corrects)
	err = []
	for i in range(len(predicts)):
		if predicts[i] == corrects[i]:
			err.append(0)
		else:
			err.append(1)
	test = np.concatenate((np.reshape(predicts,[len(predicts),1]),np.reshape(corrects,[len(corrects),1]),\
	np.reshape(err,[len(err),1])), axis=1)
	save_data = pd.DataFrame(test)
	save_data.to_csv('%s'%(mc._RESULT_FILE),index=False,header=False)
	#save_data.to_csv('../results/results.csv',index=False,header=False)

	fp = open('%s'%(mc._RESULT_FILE),'a')
	#fp = open('../results/results.csv','a')
	fp.write('%f\n'%((1.0-np.mean(err))*100.0))
	fp.close()

def transfer_model(source_df, target_df, test_df, method_flag, fold_num):

	source_labels, source_data = np.split(np.array(source_df),[1],axis=1)
	target_labels, target_data = np.split(np.array(target_df),[1],axis=1)
	test_labels, test_data = np.split(np.array(test_df),[1],axis=1)

	# normalization
	#normalized_source_data = pre.normalize(source_data)
	#normalized_target_data = pre.normalize(target_data)
	#normalized_test_data = pre.normalize(test_data)
	normalized_source_data = source_data
	normalized_target_data = target_data
	normalized_test_data = test_data


	### constuct model for source domain task  ###

	# optimization
	opt = Adam()

	# network setting
	latent = models.latent(normalized_source_data.shape[1])
	sll = models.source_last_layer()
	tll = models.target_last_layer()

	source_inputs = Input(shape=normalized_source_data.shape[1:])
	latent_features = latent(source_inputs)
	source_predictors = sll(latent_features)

	latent.trainable = mc._SORUCE_LATENT_TRAIN
	source_predictors.trainable = True

	source_nn = Model(inputs=[source_inputs], outputs=[source_predictors])
	source_nn.compile(loss=['mean_squared_error'],optimizer=opt)
	#source_nn.summary()

	# training using source domain data
	if method_flag != mc._SCRATCH:
		source_max_loop = int(normalized_source_data.shape[0]/mc._BATCH_SIZE)
		source_progbar = Progbar(target=mc._SOURCE_EPOCH_NUM)
		for epoch in range(mc._SOURCE_EPOCH_NUM):
			shuffle_data, shuffle_labels, _ = pre.paired_shuffle(normalized_source_data,source_labels,1)

			for loop in range(source_max_loop):
				batch_train_data = shuffle_data[loop*mc._BATCH_SIZE:(loop+1)*mc._BATCH_SIZE]
				batch_train_labels = shuffle_labels[loop*mc._BATCH_SIZE:(loop+1)*mc._BATCH_SIZE]
				batch_train_labels = np.reshape(batch_train_labels, [len(batch_train_labels)])
				one_hots = np.identity(mc._SOURCE_DIM_NUM)[np.array(batch_train_labels, dtype=np.int32)]
				loss = source_nn.train_on_batch([batch_train_data],[one_hots])

			#source_progbar.add(1, values=[("source loss",loss)])

		# save
		#latent.save('../results/source_latent.h5')
		#sll.save('../results/source_last_layer.h5')

	# compute relation vectors
	if method_flag == mc._SCRATCH or method_flag == mc._CONV_TRANSFER:
		target_vectors = np.identity(mc._TARGET_DIM_NUM)[np.array(target_labels, dtype=np.int32)]
		target_vectors = np.reshape(target_vectors, [target_vectors.shape[0], target_vectors.shape[2]])
	elif method_flag == mc._COUNT_ATDL:
		target_labels, relations = rv.compute_relation_labels(source_nn, normalized_target_data, target_labels, fold_num)
		target_vectors = np.identity(mc._SOURCE_DIM_NUM)[np.array(target_labels, dtype=np.int32)]
		target_vectors = np.reshape(target_vectors, [target_vectors.shape[0], target_vectors.shape[2]])
	else:
		relation_vectors = rv.compute_relation_vectors(source_nn, normalized_target_data, target_labels, fold_num, method_flag)
		target_vectors = np.zeros((len(target_labels),mc._SOURCE_DIM_NUM), dtype=np.float32)
		for i in range(len(target_labels)):
			target_vectors[i] = relation_vectors[int(target_labels[i])]

	### tuning model for target domain task	 ###

	latent.trainable = mc._TARGET_LATENT_TRAIN
	target_inputs = Input(shape=normalized_target_data.shape[1:])
	latent_features = latent(target_inputs)
	if method_flag == mc._SCRATCH or method_flag == mc._CONV_TRANSFER:
		predictors = tll(latent_features)
		label_num = mc._TARGET_DIM_NUM
	else:
		predictors= sll(latent_features)
		label_num = mc._SOURCE_DIM_NUM

	target_nn = Model(inputs=[target_inputs], outputs=[predictors])
	target_nn.compile(loss=['mean_squared_error'],optimizer=opt)
	#target_nn.summary()

	# training using target domain data
	target_max_loop = int(normalized_target_data.shape[0]/mc._BATCH_SIZE)
	target_progbar = Progbar(target=mc._TARGET_EPOCH_NUM)
	for epoch in range(mc._TARGET_EPOCH_NUM):

		shuffle_data, shuffle_labels, _ = \
		pre.paired_shuffle(normalized_target_data, target_vectors, label_num)
		for loop in range(target_max_loop):
			batch_train_data = shuffle_data[loop*mc._BATCH_SIZE:(loop+1)*mc._BATCH_SIZE]
			batch_train_labels = shuffle_labels[loop*mc._BATCH_SIZE:(loop+1)*mc._BATCH_SIZE]
			loss = target_nn.train_on_batch([batch_train_data],[batch_train_labels])
		#target_progbar.add(1, values=[("target loss",loss)])


	# compute outputs of test data of target domain
	x = target_nn.predict([normalized_test_data])
	if method_flag == mc._SCRATCH or method_flag == mc._CONV_TRANSFER:
		idx = np.argmax(x, axis=1)
	elif method_flag == mc._COUNT_ATDL:
		idx = np.argmax(x,axis=1)
		for j in range(len(test_labels)):
			for i in range(mc._TARGET_DIM_NUM):
				if test_labels[j] == i:
					test_labels[j] = relations[i]
					break
	else:
		distance, idx = Neighbors(x, relation_vectors, 1)
		idx = idx[:,0]

	backend.clear_session()
	return idx.T, test_labels.T


if __name__ == '__main__':
	method_flag = int(method_flag)
	main(method_flag)
