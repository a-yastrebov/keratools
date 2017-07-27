##################################################################
# Apply cascade classifier to test image in multiple scale
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

cls_win_width, cls_win_height = 32, 60
cc_num = 1
colorimage = False

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

def drawResp(img, resp, scale):
	for r in resp:
		rx = (int)(r[0]*scale)
		ry = (int)(r[1]*scale)
		rw = (int)((r[0]+cls_win_width)*scale)
		rh = (int)((r[1]+cls_win_height)*scale)
		cv2.rectangle(img,(rx,ry),(rw-1,rh-1),(0,0,255),1)
	
def workImage(img):
	resp = []
	if colorimage:
		h,w,c = img.shape
	else:
		h,w = img.shape
	y = 0
	
	while (y <= h - cls_win_height):
		x = 0
		while (x <= w - cls_win_width):	
			images = []
			np_images = []	
			
			img1 = img[y:y+cls_win_height, x:x+cls_win_width]				
			
			if colorimage:
				img1 = img1.transpose((2,0,1)) 
				img1 = np.expand_dims(img1, axis=0)
			else:
				img1 = img1.transpose((0,1)) 
				img1 = np.expand_dims(img1, axis=0) 
				img1 = np.expand_dims(img1, axis=0) 
	
			images.append(img1)
	
			np_images = np.vstack(images)
			res1 = model1.predict(np_images)
	
			#print("x={} y={} res={}".format(x,y,res1[0][0]))
			if (res1[0][0] > 0.9):
				res2 = model2.predict(np_images)
				if (res2[0][0] > 0.9):
					print("response found! x={} y={} res={}, {}".format(x,y,res1[0][0],res2[0][0]))
					resp.append((x,y))
			x = x + 5
		
		y = y + 5
	return resp


def workImageScaling(imagefn):
	if colorimage:
		img = cv2.imread(imagefn)
		h,w,c  = img.shape
	else:
		img = cv2.imread(imagefn,0)
		h,w  = img.shape
	img = img.astype(float)/255
	imgsrc = cv2.imread(imagefn)
	
	#print (w,h)
	
	scale = 1
	scale_step = 1.2
	h1 = (int)(h*1.0/scale)
	w1 = (int)(w*1.0/scale)
	
	while ((h1 >= cls_win_height) and (w1 >= cls_win_width)):	
		if colorimage:
			img_sc = scaleImageCol(img, 1.0/scale)
		else:
			img_sc = scaleImage(img, 1.0/scale)
		print("working image with scale={}".format(scale))
		resp = workImage(img_sc)
		drawResp(imgsrc, resp, scale)
		
		scale = scale*scale_step
		h1 = (int)(h*1.0/scale)
		w1 = (int)(w*1.0/scale)
		
	cv2.imshow("full",imgsrc)
	cv2.imwrite("out.jpg", imgsrc)

#===== main =====

# two-stage cascade hardcode
model1 = load_model('nn1.h5')
model2 = load_model('nn2.h5')

workImageScaling("test/cvn.png")
k = cv2.waitKey(0)
if (k == 27):
	exit()

#let's work set of patterns
test_path = "data/32x60cc1/validation/pos/"
files = os.listdir(test_path)
for f in files:
	print(f)
	workImageScaling(test_path+f)
	k = cv2.waitKey(0)
	if (k == 27):
		break

