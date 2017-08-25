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

head_detector_width = 32
head_detector_height = 32

img_size = w*h*3
buf_size = img_size*10
host = 'localhost'
port = 123
addr = (host,port)
patt_num=0

def classifyHead(img, headDetectorModel, clsModels):
	(rx,ry) = applySlidingDetector(img, headDetectorModel)
	
	img1 = img[ry:ry+head_detector_height, rx:rx+head_detector_width]
	cv2.imwrite('head_res.png',img1)
	
	return (rx,ry)

def applySlidingDetector(img, model):
	rx = -1
	ry = -1
	h,w,c = img.shape
	
	maxres = 0
	y = 0
	while (y <= h - head_detector_height):
		x = 0
		while (x <= w - head_detector_width):	
			images = []
			np_images = []	
			
			img1 = img[y:y+head_detector_height, x:x+head_detector_width]				
			
			img1 = img1.transpose((2,0,1)) 
			img1 = np.expand_dims(img1, axis=0)
	
			images.append(img1)
	
			np_images = np.vstack(images)
			res = model.predict(np_images)
	
			#print("x={} y={} res={}".format(x,y,res1[0][0]))
			if (res[0][0] > maxres):
				maxres = res
				rx = x
				ry = y
						
			x = x + 4		
		y = y + 4
	return (rx,ry)

#send full image in classifier
def matchPatternsSimple(img0):
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
		if r > 0.4:
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
	
	#match = matchPatternsSimple(img)
	#return match
	(rx,ry)=classifyHead(imgF, headModels, model)
	return (rx,ry)	

def matchFromBytes(buf):
	global models
	global headModel
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
	#v = matchPatternsSimple(imgF)
	#return v
	(rx,ry)=classifyHead(imgF, headModels, model)
	return (rx,ry)
	
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
hdmodel = load_model('head.h5')
models = []
models = loadModels()

matchPatternsFromFile(sys.argv[1])

quit()

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
			#v1 = matchFromBytes(img)
			#print("img done: {}".format(v1))
			#a = int(v1*100+0.5)
			#answers.append(a)
			
		b = bytes(answers)
		
		conn.send(b)		
		conn.close()
    
tcp_socket.close()


