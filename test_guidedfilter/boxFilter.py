# -*- codeing:utf8 -*-
import cv2
import numpy as np
import numpy.matlib

def boxfilter(imSrc, r):
	# r is the radius of window w 
	hei, wid = imSrc.shape
	imDst = np.zeros((imSrc.shape))
	# cumulative sum over Y axis
	imCum = np.cumsum(imSrc, axis=0)
	# difference over Y axis
	imDst[0:r+1, :] = imCum[r:2*r+1, :]
	imDst[r+1:hei-r, :] = imCum[2*r+1:hei, :] - imCum[0:hei-2*r-1, :]
	imDst[hei-r:hei, :] = numpy.matlib.repmat(imCum[hei-1, :], r, 1)\
		- imCum[hei-2*r-1:hei-r-1, :]

	# cumulative sum over X axis
	imCum = np.cumsum(imDst, axis = 1)
	# difference over Y axis
	imDst[:, 0:r+1] = imCum[:, r:2*r+1]
	imDst[:, r+1:wid-r] = imCum[:, 2*r+1:wid] - imCum[:, 0:wid-2*r-1]
	imDst[:, wid-r:wid] = numpy.matlib.repmat(np.matrix(imCum[:, wid-1]).T, 1, r)\
		- imCum[:, wid-2*r-1:wid-r-1]
	return imDst

