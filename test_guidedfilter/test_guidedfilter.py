import numpy as np
import cv2
import guidedFilter as gf
import scipy.io as sio


img = cv2.imread('./img_enhancement/tulips.bmp') / 255.0;
test_img = img[:, :, 2];
p = test_img;

r = 16;
eps = 0.01; 

q = np.zeros(test_img.shape);

# q,N,mean_I, mean_p, mean_Ip, cov_Ip,mean_II, var_I, a, b, mean_a, mean_b, q = gf.guidedfilter(test_img, p, r, eps);
q = gf.guidedfilter(test_img, p, r, eps);

I_enhanced = (test_img - q) * 5 + q;

# plt.subplot(1,3,1), plt.imshow(test_img)
# # plt.subplot(132), plt.imshow(q)
# # plt.subplot(133), plt.imshow(I_enhanced)

# # plt.show()

cv2.imshow('adsdfd', test_img)
cv2.imshow('ada', q)
cv2.imshow('sdfda', I_enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()
# sio.savemat('saveddata.mat', {'N': N,'mean_I': mean_I, 'mean_p': mean_p, 'mean_Ip': mean_Ip, 'cov_Ip': cov_Ip, \
# 	'mean_II':mean_II, 'var_I':var_I,'a':a, 'b':b, 'mean_a': mean_a, 'mean_b': mean_b, 'q': q, 'I_enhanced': I_enhanced})
