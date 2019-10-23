import os
import re
import sys

import numpy as np
import pandas as pd
import collections
from scipy.stats import multivariate_normal as mn

# Please modify to fit your environment
from tensorflow.contrib.keras.api.keras import backend as K
import tensorflow.contrib.keras.api.keras as keras
from tensorflow.contrib.keras.api.keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.layers import add, Input, Dense, Dropout
from sklearn.covariance import graph_lasso, GraphLasso, GraphLassoCV, empirical_covariance

import macro as mc

def modify_relation_vectors(init_vectors, predict_vectors, labels):

	df_predict_vectors = pd.DataFrame(predict_vectors)
	df_predict_vectors['lab'] = labels

	covs = np.zeros((mc._TARGET_PROPERTY_NUM, mc._SOURCE_PROPERTY_NUM,mc._SOURCE_PROPERTY_NUM), dtype=np.float32)
	invs = np.zeros((mc._TARGET_PROPERTY_NUM, mc._SOURCE_PROPERTY_NUM,mc._SOURCE_PROPERTY_NUM), dtype=np.float32)
	for j in range(mc._TARGET_PROPERTY_NUM):
		bak = df_predict_vectors[df_predict_vectors['lab']==j]
		bak = bak.drop('lab', axis=1)
		covs[j] = np.cov(bak.values.T)

	for i in range(mc._TARGET_PROPERTY_NUM):
		invs[i] = np.linalg.inv(covs[i])
		#covs[i], invs[i], _ = graph_lasso(covs[i], alpha=0.001, max_iter=1000, return_costs=True)


	relation_vectors = init_vectors
	best_y = 0.0
	for j in range(mc._TARGET_PROPERTY_NUM):
		for k in range(mc._TARGET_PROPERTY_NUM):
			if j != k:
				sub = init_vectors[j]-init_vectors[k]
				best_y = best_y-1.0/(np.linalg.norm(sub)+mc._EPS)

	for i in range(mc._ITERATION):
		for j in range(mc._TARGET_PROPERTY_NUM):
			x_star = np.random.multivariate_normal(init_vectors[j], covs[j], mc._SAMPLING)
			sub = x_star-init_vectors[j]
			sub_cov = np.dot(sub,invs[j])
			y = -np.sum(sub_cov*sub,axis=1)

			for k in range(mc._TARGET_PROPERTY_NUM):
				if j != k:
					sub = x_star-init_vectors[k]
					y = y-1.0/(np.linalg.norm(sub)+mc._EPS)

			min_y = np.min(y)
			if best_y > min_y:
				best_y = min_y
				relation_vectors[j] = x_star[np.argmin(y)]

	return relation_vectors

def compute_relation_vectors(nn, data, labels, fold_num, method_flag):

	predict_vectors = np.array(nn.predict(data))
	relation_vectors = np.zeros((mc._TARGET_PROPERTY_NUM,mc._SOURCE_PROPERTY_NUM), dtype=np.float32)
	lab_num = np.zeros(mc._TARGET_PROPERTY_NUM, dtype=np.int32)
	for i in range(len(data)):
		for j in range(mc._TARGET_PROPERTY_NUM):
			if labels[i] == j:
				relation_vectors[j] = relation_vectors[j] + predict_vectors[i]
				lab_num[j] = lab_num[j]+1

	for j in range(mc._TARGET_PROPERTY_NUM):
		relation_vectors[j] = relation_vectors[j]/lab_num[j]

	if fold_num == 1 and method_flag == mc._MODIFIED_MEAN_ATDL:
		save_data = pd.DataFrame(relation_vectors)
		save_data.to_csv('../results/relation_vectors_before.csv',index=False,header=False)

	if method_flag == mc._MODIFIED_MEAN_ATDL:
		relation_vectors = modify_relation_vectors(relation_vectors, predict_vectors, labels)

	if fold_num == 1 and method_flag == mc._MODIFIED_MEAN_ATDL:
		save_data = pd.DataFrame(relation_vectors)
		save_data.to_csv('../results/relation_vectors_after.csv',index=False,header=False)

	return relation_vectors


def compute_relation_labels(nn, data, labels, fold_num):

	predict_vectors = np.array(nn.predict(data))
	df_predict_vectors = pd.DataFrame(predict_vectors)
	df_predict_vectors['lab'] = labels
	relations = []
	for j in range(mc._TARGET_PROPERTY_NUM):
		bak = df_predict_vectors[df_predict_vectors['lab']==j]
		bak = bak.drop('lab', axis=1)
		for i in range(len(relations)):
			bak = bak.drop(relations[i], axis=1)
		idx = np.argmax(bak.values, axis=1)
		cidx = collections.Counter(idx)
		relations.append(cidx.most_common()[0][0])
	for j in range(len(labels)):
		for i in range(mc._TARGET_PROPERTY_NUM):
			if labels[j] == i:
				labels[j] = relations[i]
				break

	if fold_num == 1:
		cidxs = np.zeros((mc._TARGET_PROPERTY_NUM,mc._SOURCE_PROPERTY_NUM), dtype=np.int32)
		for j in range(mc._TARGET_PROPERTY_NUM):
			bak = df_predict_vectors[df_predict_vectors['lab']==j]
			bak = bak.drop('lab', axis=1)
			idx = np.argmax(bak.values, axis=1)
			cidx = collections.Counter(idx)
			for i in range(mc._SOURCE_PROPERTY_NUM):
				cidxs[j][i] = cidx[i]

		save_data = pd.DataFrame(cidxs)
		save_data.to_csv('../results/count_ver_relation_vectors.csv',index=False,header=False)

	return labels, relations
