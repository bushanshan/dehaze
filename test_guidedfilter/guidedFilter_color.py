# -*- coding:utf8 -*-
import cv2
import numpy as np
import boxFilter as bF
from numpy.linalg import inv

def guidedfilter_color(I, p, r, eps):
	# guidance image: I (should be a color (BGR) image)
	# filtering input image: p (should be a gray-scale/single channel image)
	# local window radius: r
	# regularization parameterL epsilon

	hei, wid = p.shape
	# the size of each local patch N=(2r+1)^2 except for boundary pixels
	N = bF.boxfilter(np.ones((hei, wid)), r)

	mean_I_r = bF.boxfilter(I[:, :, 2], r) / N
	mean_I_b = bF.boxfilter(I[:, :, 0], r) / N
	mean_I_g = bF.boxfilter(I[:, :, 1], r) / N

	mean_p = bF.boxfilter(p, r) / N

	mean_Ip_r = bF.boxfilter(I[:, :, 2] * p, r) / N
	mean_Ip_b = bF.boxfilter(I[:, :, 0] * p, r) / N
	mean_Ip_g = bF.boxfilter(I[:, :, 1] * p, r) / N
	

	# covariance of (I, p) in each local patch
	cov_Ip_b = mean_Ip_b - mean_I_b * mean_p
	cov_Ip_g = mean_Ip_g - mean_I_g * mean_p
	cov_Ip_r = mean_Ip_r - mean_I_r * mean_p

	# variance of I in each local patch: the matrix Sigma in Eqn(14)
	# Note the variance in each local patch is a 3*3 symmetric matrix
	#  			rr, rg, rb
	# Sigma = 	rg, gg, gb
	# 			rb, gb, bb
	var_I_rr = bF.boxfilter(I[:, :, 2] * I[:, :, 2], r) / N - mean_I_r * mean_I_r
	var_I_rg = bF.boxfilter(I[:, :, 2] * I[:, :, 1], r) / N - mean_I_r * mean_I_g
	var_I_rb = bF.boxfilter(I[:, :, 2] * I[:, :, 0], r) / N - mean_I_r * mean_I_b
	var_I_gg = bF.boxfilter(I[:, :, 1] * I[:, :, 1], r) / N - mean_I_g * mean_I_g
	var_I_gb = bF.boxfilter(I[:, :, 1] * I[:, :, 0], r) / N - mean_I_g * mean_I_b
	var_I_bb = bF.boxfilter(I[:, :, 0] * I[:, :, 0], r) / N - mean_I_b * mean_I_b

	a = np.zeros((hei, wid, 3))
	for y in range(hei):
		for x in range(wid):
			Sigma = np.matrix([[var_I_rr[y, x], var_I_rg[y, x], var_I_rb[y, x]],
				[var_I_rg[y, x], var_I_gg[y, x], var_I_gb[y, x]],
				[var_I_rb[y, x], var_I_gb[y, x], var_I_bb[y, x]]])
			# Sigma = Sigma + eps *  eys(3)

			cov_Ip = np.matrix([cov_Ip_r[y, x], cov_Ip_g[y, x], cov_Ip_b[y, x]])

			# Eqn(14) int the paper
			a[y, x, :] = cov_Ip * inv(Sigma + eps * np.eye(3))

	b = mean_p - a[:, :, 0] * mean_I_r - a[:, :, 1] * mean_I_g - a[:, :, 2] * mean_I_b

	# Eqn(16) in the paper
	b1 = bF.boxfilter(a[:, :, 0], r) * I[:, :, 2]
	q = (bF.boxfilter(a[:, :, 0], r) * I[:, :, 2]\
		+ bF.boxfilter(a[:, :, 1], r) * I[:, :, 1]\
		+ bF.boxfilter(a[:, :, 2], r) * I[:, :, 0]\
		+ bF.boxfilter(b, r)) / N

	# return q, N,  mean_I_b,mean_I_g, mean_I_r,mean_p,mean_Ip_b,mean_Ip_g,mean_Ip_r,cov_Ip_b,cov_Ip_g,cov_Ip_r,var_I_bb,var_I_gb,\
	# 	var_I_rb,var_I_gg,var_I_rg,var_I_rr,a,b,q, b1, cov_Ip
	return q 







