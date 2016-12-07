import dicom
import pylab
import matplotlib.pyplot as plt

import numpy as np
import cv2

ds=dicom.read_file("../RSI_Images/7.dcm")
pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone,interpolation='nearest')
plt.axis('off')
plt.savefig('test.png',bbox_inches='tight', pad_inches=0);

img = cv2.imread('test.png')

dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
plt.subplot(121),plt.imshow(img)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (255,255,255), 3)
cv2.imshow("Image Contours",img)
cv2.waitKey(0)
cv2.destroyAllWindows()



