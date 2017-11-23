##################################################################
# Match two patterns in one two-pattern image
# Works for color images only
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

cls_win_width, cls_win_height = 32, 60

def matchPatterns(img):
	h,w,c = img.shape
	
	images = []
	np_images = []	
	
	#img = img.transpose((2,0,1))
	
	
	#for j in range(0,h):
	#	for i in range(0,w):
	#		print(img[j][i][0],img[j][i][1],img[j][i][2])
	
	img1 = np.expand_dims(img, axis=0) 
	
	images.append(img1)
	
	np_images = np.vstack(images)
	res = model.predict(np_images)
	
	#print("response: {}".format(res[0][0]))
	
	return res[0][0]


def matchPatternsFromFile(imagefn):
	img = cv2.imread(imagefn,)
	img = cv2.resize(img,(cls_win_width,cls_win_height))
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img = img.astype(float)/255	
	
	match = matchPatterns(img)
	return match	
	
def matchPatternsFromDir(dir):
	fns= os.listdir(dir)
	
	print("")
	for fn in fns:
		if fn.endswith(".png") or fn.endswith(".bmp") or fn.endswith(".jpg"):
			v = matchPatternsFromFile(dir+"/"+fn)
			print("{} {}".format(fn,v)) 

#===== main =====

model = load_model(sys.argv[1])
match = matchPatternsFromDir(sys.argv[2])

