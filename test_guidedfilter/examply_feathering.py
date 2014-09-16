import cv2
import numpy as np
import guidedFilter_color as gfc
import scipy.io as sio


I = cv2.imread('toy.bmp') / 255.0
toy_mask = cv2.imread('toy-mask.bmp')
gray = cv2.cvtColor(toy_mask, cv2.COLOR_BGR2GRAY)
p = gray / 255.0
# p = double(rgb2gray(imread('toy-mask.bmp'))) / 255;

r = 60;
eps = 0.000001;
print eps

q, N,  mean_I_b,mean_I_g, mean_I_r,mean_p,mean_Ip_b,mean_Ip_g,mean_Ip_r,cov_Ip_b,cov_Ip_g,cov_Ip_r,\
	var_I_bb,var_I_gb,var_I_rb,var_I_gg,var_I_rg,var_I_rr,a,b,q, b1, cov_Ip = gfc.guidedfilter_color(I, p, r, eps);

# figure();
# imshow([I, repmat(p, [1, 1, 3]), repmat(q, [1, 1, 3])], [0, 1]);
cv2.imshow('a',I)
cv2.imshow('c',p)
cv2.imshow('b',q)

cv2.waitKey(0)
cv2.destroyAllWindows()

sio.savemat('saveddata.mat', {'N': N,'mean_I_b': mean_I_b, 'mean_I_g': mean_I_g, 'mean_I_r':mean_I_r, 'mean_p': mean_p,\
 	'mean_Ip_b': mean_Ip_b,'mean_Ip_g': mean_Ip_g,'mean_Ip_r': mean_Ip_r, 'cov_Ip_b': cov_Ip_b,'cov_Ip_g': cov_Ip_g,'cov_Ip_r': cov_Ip_r,\
	'var_I_bb':var_I_bb,'var_I_gb':var_I_gb,'var_I_rb':var_I_rb,'var_I_gg':var_I_gg,'var_I_rg':var_I_rg,'var_I_rr':var_I_rr,\
	'a':a,'a1': a[:,:,0], 'a2':a[:,:,1], 'a3':a[:,:,2], 'b':b,'q': q, 'b1':b1, 'cov_Ip':cov_Ip})
