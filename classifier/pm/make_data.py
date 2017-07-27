##################################################################
# Make a dataset of two-patterns images
# 
# Copyright (c) 2017 Alexey Yastrebov
# MIT License, see LICENSE file.
##################################################################

#coding=utf-8
import numpy as np
import cv2
import os
import sys
import random
import imgbase

cls_win_width, cls_win_height = 32, 60
true_data_dir = "truedata"
false_data_dir = "falsedata"
max_patts_per_case = 110

def joinImageHor(img1, img2, pad):
		"""Join two images side by side with pad black pixels between images"""
		
		h,w,c = img1.shape
		img_res = np.zeros((h,w*2+pad,3), np.uint8)
		img_res[0:h, 0:w] = img1[0:h, 0:w]
		img_res[0:h, w+pad:2*w+pad] = img2[0:h, 0:w]
		
		return img_res

def truncToMaxCount(patts):
	if len(patts) <= max_patts_per_case:
		return
	while len(patts) > max_patts_per_case:
		i = random.randint(0,len(patts)-1)
		patts.pop(i)

def makeDataFromCase(path):
	case_subdirs = os.listdir(path)
	true_data_path = path+"/"+true_data_dir
	false_data_path = path+"/"+false_data_dir
	true_fns1 = os.listdir(true_data_path) #file names with similar patterns
	false_fns1 = os.listdir(false_data_path) #file names with dissimilar patterns
	
	true_fns = []
	for fn in true_fns1:
		if fn.endswith(".png") or fn.endswith(".bmp") or fn.endswith(".jpg"):
			true_fns.append(true_data_path+"/"+fn)
			
	false_fns = []
	for fn in false_fns1:
		if fn.endswith(".png") or fn.endswith(".bmp") or fn.endswith(".jpg"):
			false_fns.append(false_data_path+"/"+fn)
	
	true_imgs = []
	for fn in true_fns:
		img = cv2.imread(fn)
		img = cv2.resize(img,(cls_win_width,cls_win_height))
		true_imgs.append(img)
	false_imgs = []
	for fn in false_fns:
		img = cv2.imread(fn)
		img = cv2.resize(img,(cls_win_width,cls_win_height))
		false_imgs.append(img)
			
	#make pairs of similar patterns
	pos_imgs_res = []
	for i in range(len(true_imgs)):
		img1 = true_imgs[i]
		for j in range(len(true_imgs)):
			img2 = true_imgs[j]
			img_res = imgbase.joinImageHor(img1,img2,3)
			pos_imgs_res.append(img_res)
			if i != j:
				img_res = imgbase.joinImageHor(img2,img1,3)
				pos_imgs_res.append(img_res)
	
	#make pairs of pos+neg patterns
	n=0
	neg_imgs_res = []
	for i in range(len(true_imgs)):
		img1 = true_imgs[i]
		for j in range(len(false_imgs)):
			img2 = false_imgs[j]
			if n % 2 == 0:
				img_res = imgbase.joinImageHor(img1,img2,3)
			else:
				img_res = imgbase.joinImageHor(img2,img1,3)
			neg_imgs_res.append(img_res)
			n=n+1
	
	truncToMaxCount(pos_imgs_res)
	truncToMaxCount(neg_imgs_res)
	return pos_imgs_res,neg_imgs_res
		
def saveData(pos_res, neg_res, out_path):
	try:
		os.mkdir(out_path)
		os.mkdir(out_path+"/pos")
		os.mkdir(out_path+"/neg")
	except:
		print()
	n = 1
	for img in pos_res:
		cv2.imwrite(out_path+"/pos/pos{}.png".format(n),img)
		n = n+1
	n = 1
	for img in neg_res:
		cv2.imwrite(out_path+"/neg/neg{}.png".format(n),img)
		n = n+1
	
def makeData(path):
	pos = []
	neg = []
	
	case_paths = os.listdir(path)
	for p in case_paths:
		pos1 = []
		neg1 = []
		print("workng "+path+"/"+p)
		pos1,neg1 = makeDataFromCase(path+"/"+p)
		pos.extend(pos1)
		neg.extend(neg1)
	return pos,neg			

random.seed()
in_path = sys.argv[1] #source images path
out_path = sys.argv[2] #out dataset path	
pos_res,neg_res = makeData(in_path)
saveData(pos_res,neg_res,out_path)