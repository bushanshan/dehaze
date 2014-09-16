import cv2
import numpy as np
import guidedFilter as gf


I = cv2.imread('./cat.bmp') 
gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
# print gray.shape
gray = I[:, :, 0] / 255.0
p = gray

r = 4 
# try r=2, 4, or 8
eps = 0.04; 
# % try eps=0.1^2, 0.2^2, 0.4^2

q = gf.guidedfilter(gray, p, r, eps);


cv2.imshow('a',I)
cv2.imshow('b',q)
cv2.waitKey(0)
cv2.destroyAllWindows()