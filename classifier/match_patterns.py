##################################################################
# Match two patterns in one two-pattern image
# Works only for color images
# 
# Copyright (c) 2017 Alexey Yastrebov
# MIT License, see LICENSE file.
##################################################################

#coding=cp1251
from keras.models import load_model
from keras.preprocessing import image as image_utils
import numpy as np
import cv2
import os
import sys

def matchPatterns(img):
	h,w,c = img.shape
	
	images = []
	np_images = []	
	
	img1 = img.transpose((2,0,1)) 
	img1 = np.expand_dims(img1, axis=0) 
	
	images.append(img1)
	
	np_images = np.vstack(images)
	res = model.predict(np_images)
	
	#print("response: {}".format(res[0][0]))
	
	return res[0][0]


def matchPatternsFromFile(imagefn):
	img = cv2.imread(imagefn)
	img = cv2.resize(img,(30,60))
	img = img.astype(float)/255	
	
	match = matchPatterns(img)
	return match	
	
def matchPatternsFromDir(dir):
	fns= os.listdir(dir) #file names with dissimilar patterns	
	
	for fn in fns:
		if fn.endswith(".png") or fn.endswith(".bmp") or fn.endswith(".jpg"):
			v = matchPatternsFromFile(dir+"/"+fn)
			print("{} {}".format(fn,v)) 

#===== main =====

model = load_model('nn00000.h5')
match = matchPatternsFromDir(sys.argv[1])

