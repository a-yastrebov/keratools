##################################################################
# TCP analytics server
# Receives images and sends classifier response
# Works with color images only
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
from socket import *

h = 60
w = 32 #67

img_size = w*h*3
buf_size = img_size*10
host = 'localhost'
port = 123
addr = (host,port)
patt_num=0

def matchPatterns(img0):
	h,w,c = img0.shape
	global patt_num
	global models
	
	images = []
	np_images = []
	results = []
	
	patt_num = patt_num + 1
	
	img1 = img0.transpose((2,0,1)) 
	img1 = np.expand_dims(img1, axis=0) 
	
	images.append(img1)
	
	np_images = np.vstack(images)
	for model in models:
		res = model.predict(np_images)
		results.append(res[0][0])
	
	n = 0
	for r in results:
		if r > 0.5:
			n = n+1
	if len(results) == 0:
		answer = 0
	else:
		answer = n/len(results)
	
	#print("response: {} {} {}".format(res1[0][0],res2[0][0],res3[0][0]))
	print("answer: {}".format(answer))
	
	return answer


def matchPatternsFromFile(imagefn):
	img = cv2.imread(imagefn)
	img = img.astype(float)/255	
	
	match = matchPatterns(img)
	return match	

def matchFromBytes(buf):
	global patt_num
	if len(buf) != img_size:
		return 0
	img = np.zeros((h,w,3), np.uint8)
	n = img_size
	i = 0
	for y in range(h) :
		for x in range(w) :
			for c in range(3) :
				img[y,x,c] = buf[i]
				i = i+1
		
	imgF = img.astype(float)/255.0	 
	v = matchPatterns(imgF)
	return v
	
def loadModels():
	modles = []
	print("Building forest...")
	files = os.listdir("./")
	for fn in files:
		if fn.endswith(".h5"):
			print(fn)
			model = load_model(fn)
			models.append(model)
	
	return models
	
def splitData(data):
	global w
	global h
	imgs = []
	bufSize = len(data)
	imgSize = w*h*3
	i = 0
	while i < bufSize:
		imgs.append(data[i:i+imgSize])
		i = i+imgSize
	
	return imgs
	
	
#===== main =====
#model1 = load_model('nn1.h5')
models = []
models = loadModels()

tcp_socket = socket(AF_INET, SOCK_STREAM)
tcp_socket.bind(addr)
tcp_socket.listen(1)

print("starting server...")
while True:
	print('wait connection...')
	conn, addr = tcp_socket.accept()
	print('client addr: ', addr)
	
	data = conn.recv(buf_size)
	
	if not data:
		conn.close()
		break
	else:
		imgs = splitData(data)
		answers = []
		for img in imgs:			
			v1 = matchFromBytes(img)
			print("img done: {}".format(v1))
			a = int(v1*100+0.5)
			answers.append(a)
			
		b = bytes(answers)
		
		conn.send(b)		
		conn.close()
    
tcp_socket.close()


