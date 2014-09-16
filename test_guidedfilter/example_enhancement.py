 # example: detail enhancement
 # figure 6 in our paper
import cv2
import numpy as np
import guidedFilter as gf


I = (cv2.imread('/Users/bushanshan/Documents/Workspace/haze_removal/myCode/img_enhancement/tulips.bmp')) / 255.0;
p = I;
r = 16;
eps = 0.01;

q = np.zeros(I.shape);

q[:, :, 0] = gf.guidedfilter(I[:, :, 2], p[:, :, 0], r, eps);
q[:, :, 1] = gf.guidedfilter(I[:, :, 1], p[:, :, 1], r, eps);
q[:, :, 2] = gf.guidedfilter(I[:, :, 0], p[:, :, 2], r, eps);

I_enhanced = (I - q) * 5 + q;

print q.shape
print I.shape
print I_enhanced.shape


cv2.imshow("I", I)
cv2.imshow("q", q)
cv2.imshow('I_enhanced', I_enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()
# figure();
# imshow([I, q, I_enhanced], [0, 1]);