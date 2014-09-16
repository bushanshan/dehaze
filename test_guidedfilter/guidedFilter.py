# -*- coding:utf8 -*-
import cv2
import numpy as np
import boxFilter as bF

def guidedfilter(I, p, r, eps):
	# guidance image: I (should be a gray-scale/single channel image)
	# filtering imput image: p (should be a gray-scale/single channel image)
	# local window radius: r
	# regularization parameter:eps
	hei, wid = I.shape[: 2]
	# the size of each local patch N=(2r+1)^2 except for boundary pixels
	N = bF.boxfilter(np.ones((hei, wid)), r)

	mean_I = bF.boxfilter(I, r) / N

	mean_p = bF.boxfilter(p, r) / N
	mean_Ip = bF.boxfilter(I * p, r) / N
	# this is the covariance of (I, p) in each local patch
	cov_Ip = mean_Ip - mean_I * mean_p

	mean_II = bF.boxfilter( I * I, r) / N
	var_I = mean_II - mean_I * mean_I

	# Eqn(5) in the paper
	a = cov_Ip / (var_I + eps)
	# Eqn(6) in the paper
	b = mean_p - a * mean_I

	mean_a = bF.boxfilter(a, r) / N
	mean_b = bF.boxfilter(b, r) / N

	# Eqn(8) in the paper
	q = mean_a * I + mean_b
	# return q, N, mean_I, mean_p, mean_Ip, cov_Ip, mean_II, var_I, a, b, mean_a, mean_b, q
	return q