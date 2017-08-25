##################################################################
# Make negative patterns for dataset with existing detection cascade
# 
# Copyright (c) 2017 Alexey Yastrebov
# MIT License, see LICENSE file.
##################################################################

#coding=utf-8
from keras.models import load_model
from keras.preprocessing import image as image_utils
import numpy as np
import cv2
import os
import sys
import random

cls_win_width, cls_win_height = 24, 24 
n = 0
colorimage = False # true for 3-channels models, false for 1-channel model

def scaleImageInternal(src, w, h, scale):	
	h1 = (int)(h*scale)
	w1 = (int)(w*scale)
	dst = cv2.resize(src,(w1,h1))
	return dst

def scaleImage(src, scale):
	h,w = src.shape	
	dst = scaleImageInternal(src,w,h,scale)
	return dst
	
def scaleImageCol(src, scale):
	h,w,c = src.shape
	dst = scaleImageInternal(src,w,h,scale)
	return dst
	
def workImage(img, imgsrc):
	global n
	if colorimage:
		h,w,c = img.shape
	else:
		h,w = img.shape
	y = 0
	
	while (y <= h/2 - cls_win_height):
		x = 0
		while (x <= w - cls_win_width):	
			images = []
			np_images = []	
			
			img1 = img[y:y+cls_win_height, x:x+cls_win_width]
			
			#cv2.imshow("part",img1)
			#k = cv2.waitKey(0)
			#if (k == 27):
			#	exit()
			
			if colorimage:
				img1 = img1.transpose((2,0,1)) 
				img1 = np.expand_dims(img1, axis=0) 
			else:
				img1 = img1.transpose((0,1)) 
				img1 = np.expand_dims(img1, axis=0) 
				img1 = np.expand_dims(img1, axis=0) 
	
			images.append(img1)
	
			#np_images = np.vstack(images)
			#res = model.predict(np_images)
			
			# positive answer of existing model (cascade) on GT condition negative data is negative pattern for next classifier in cascade	
			#if (res[0][0] > 0.5):
			#	n = n+1
			#	print("positive response found! x={} y={} res={}".format(x,y,res[0][0]))
			#	imgres = imgsrc[y:y+cls_win_height, x:x+cls_win_width]
			#	cv2.imwrite("neg/neg{}.bmp".format(n), imgres)
			i = random.randint(0,9)
			if i <= 3:
				n = n+1
				imgres = imgsrc[y:y+cls_win_height, x:x+cls_win_width]
				cv2.imwrite("neg/neg{}.png".format(n), imgres)
			x = x + 10
		
		y = y + 10
	
def workImageScaling(imagefn):
	if colorimage:
		img = cv2.imread(imagefn)
		h,w,c  = img.shape
	else:
		img = cv2.imread(imagefn,0)
		h,w  = img.shape
	img = img.astype(float)/255
	imgsrc = cv2.imread(imagefn)
		
	scale = 1
	scale_step = 1.2
	h1 = (int)(h*1.0/scale)
	w1 = (int)(w*1.0/scale)
	
	#while ((h1 >= cls_win_height) and (w1 >= cls_win_width)):
	if colorimage:
		img_sc = scaleImageCol(img, 1.0/scale)
	else:
		img_sc = scaleImage(img, 1.0/scale)
	imgsrc_sc = scaleImageCol(imgsrc, 1.0/scale)
	print("working image with scale={}".format(scale))
	workImage(img_sc, imgsrc_sc)		
	
	scale = scale*scale_step
	h1 = (int)(h*1.0/scale)
	w1 = (int)(w*1.0/scale)
		
#===== main =====
#model = load_model('nn1.h5')

#input data is the large images without classifying objects
data_path = sys.argv[1]
files = os.listdir(data_path)
for f in files:
	print(f)
	workImageScaling(data_path+'\\'+f)
	k = cv2.waitKey(0)
	if (k == 27):
		break
		
print("Done")
