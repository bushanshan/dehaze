import cv2
import numpy as np
import guidedFilter as gf


I = cv2.imread('cave-flash.bmp') / 255.0;
p = cv2.imread('cave-noflash.bmp') / 255.0;

r = 8;
eps = 0.0004

q = np.zeros(I.shape);

q[:, :, 0] = gf.guidedfilter(I[:, :, 2], p[:, :, 0], r, eps);
q[:, :, 1] = gf.guidedfilter(I[:, :, 1], p[:, :, 1], r, eps);
q[:, :, 2] = gf.guidedfilter(I[:, :, 0], p[:, :, 2], r, eps);

# figure();
# imshow([I, p, q], [0, 1]);
cv2.imshow('I', I)
cv2.imshow("p", p)
cv2.imshow('q', q)
cv2.waitKey(0)
cv2.destroyAllWindows()