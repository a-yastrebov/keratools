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

h = 0
w = 0

objNum = 0

def loadDataSetParams(path):
	global w,h
	f = open(path)
	lines = []
	for line in f:
		lines.append(line)
	
	if len(lines) >= 2:
		w = int(lines[0])
		h = int(lines[1])

def classifyByStage(stage, np_images):
	models = stage[0]
	clsThr = stage[1]
	voteThr = stage[2]
	results = []

	#f = open(working_dir+"/res/{number:05}ans.txt".format(number=objNum),"w")
	
	for model in models:
		res = model.predict(np_images)
		results.append(res[0][0])
		
		#debug
		#for a in res:
		#	for b in a:
		#		f.write("  {}".format(b))
		#	f.write("\n")
	#f.close()
	
	n = 0
	for r in results:
		print(r)
		if r > clsThr:
			n = n+1
	if len(results) == 0:
		answer = 0
	else:
		answer = n/len(results)
	print("stage result: {}/{}".format(n,len(results)))
		
	if answer > voteThr:
		return True
	else:
		return False
		
def classifyByCascade(cascade, img0):
	h,w,c = img0.shape
	global patt_num
		
	images = []
	np_images = []
	results = []
	
	patt_num = patt_num + 1
	
	img1 = img0.transpose((2,0,1)) 
	img1 = np.expand_dims(img1, axis=0) 
	
	images.append(img1)
	
	np_images = np.vstack(images)
	
	answer = 0
	
	for st in cascade:
		res = classifyByStage(st, np_images)
		if res == True:
			answer = 1
			break
	
	#print("response: {} {} {}".format(res1[0][0],res2[0][0],res3[0][0]))
	print("answer: {}".format(answer))
	
	return answer


def matchPatternsFromFile(imagefn):
	global cascade
	img = cv2.imread(imagefn)
	img = img.astype(float)/255	
	
	match = classifyByCascade(cascade, img)
	return match	

def matchFromBytes(cascade, buf):
	global objNum
	objNum = objNum+1
	
	if len(buf) != img_size:
		print("Error! Wrong img buf len: {}".format(len(buf)))
		return 0
	img = np.zeros((h,w,3), np.uint8)
	n = img_size
	i = 0
	for y in range(h) :
		for x in range(w) :
			for c in range(3) :
				img[y,x,c] = buf[i]
				i = i+1

	#cv2.imwrite(working_dir+"/res/{number:05}img.png".format(number=objNum),img)		
	imgF = img.astype(float)/255.0	 
	v = classifyByCascade(cascade, imgF)
	return v
	
def loadCascade(path):
	cascade = []
	print("cascade path "+path)
	subdirs = os.listdir(path)	
	for sd in subdirs:
		stage = loadStage(cls_dir+"/"+sd)
		cascade.append(stage)	
	return cascade

def loadStage(path):
	print("Load cascade stage {}".format(path))
	models = loadModels(path)
	voteThr = 0 # если больше - ступень сработала на отбрасывание
	clsThr = 0 # если больше - классификатор сработал на отбрасывание	
	
	f = open(path+"/stageparams.txt")
	lines = []
	for line in f:
		lines.append(line)
	
	if len(lines) >= 2:
		clsThr = float(lines[0])
		voteThr = float(lines[1])
	
	print("clsThr: {}".format(clsThr))
	print("voteThr: {}".format(voteThr))
	stage = []
	stage.append(models)
	stage.append(clsThr)
	stage.append(voteThr)
	return stage

def loadModels(path):
	models = []
	print("Load forest from "+path)
	files = os.listdir(path)
	for fn in files:
		if fn.endswith(".h5"):
			print(fn)
			model = load_model(path+"/"+fn)
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
working_dir = "AnalyticsServer"
cls_dir = working_dir+"/cls"

cascade = []
loadDataSetParams(working_dir+"/ds.conf")
cascade = loadCascade(cls_dir)

img_size = w*h*3
buf_size = img_size*10
host = 'localhost'
port = 3123
addr = (host,port)
patt_num=0

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
			v1 = matchFromBytes(cascade, img)
			print("img done: {}".format(v1))
			a = int(v1*100+0.5)
			answers.append(a)
			
		b = bytes(answers)
		
		conn.send(b)		
		conn.close()
    
tcp_socket.close()


