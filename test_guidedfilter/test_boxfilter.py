import cv2
import numpy as np
import scipy.io as sio
import boxFilter as bf


img = cv2.imread('./img_enhancement/tulips.bmp') / 255.0
test_img = img[:, :, 0]

r = 16

imDst = bf.boxfilter(test_img, r)

sio.savemat('saveddata.mat', {'imDst_py': imDst, 'test_img': test_img})

