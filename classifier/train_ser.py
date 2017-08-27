##################################################################
# Train CNN classifiers with presetted or random structure
# Example: train_ser.py <dataset_dir>
# 
# Copyright (c) 2017 Alexey Yastrebov
# MIT License, see LICENSE file.
##################################################################

from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Convolution2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K
from keras import callbacks
from keras.utils import plot_model
import sys
import os
import random

cls_win_width, cls_win_height = 0,0 

def getSamplesCount(path):
	cnt = 0
	filenames = os.listdir(path)
	for fn in filenames:
		if fn.endswith(".png") or fn.endswith(".bmp") or fn.endswith(".jpg"):
			cnt = cnt + 1
	return cnt

def loadDataSetParams(path):
	global cls_win_width,cls_win_height
	f = open(path)
	lines = []
	for line in f:
		lines.append(line)
	
	if len(lines) >= 2:
		cls_win_width = int(lines[0])
		cls_win_height = int(lines[1])
	
def logStruct(cnn_struct, full_con_cnt, m_num):
	inp_shape=(3, cls_win_height, cls_win_width)
	f = open(root_data_dir+"/res/struct{number:05}.txt".format(number=m_num),"w")
	k = 1
	for layer in cnn_struct:
		f_cnt = layer[0]
		drop = layer[1]
		pooling = layer[2]
		f.write("cnnlayer{}: f={} d={} p={}\n".format(k,f_cnt,drop,pooling))
		k = k+1
	f.write("full_cnt={}".format(full_con_cnt))
	f.close()
	
def randomInit():
	cnn_layers_cnt = 2#random.randint(2,5)
	struct = []
		
	for k in range(cnn_layers_cnt):
		layer = []
		layer.append(random.randint(4,20)) #filterscnt
		
		v = random.randint(0,9)
		dropout = 0
		if v >= 0:
			dropout = random.uniform(0.2, 0.5)
		layer.append(dropout) #dropout
		
		v = random.randint(0,9)
		pooling = 0
		if v >= 4:
			pooling = 1
		layer.append(pooling)#pooling2x2
		
		struct.append(layer)
		
	fullconcnt = random.randint(10,50)
	return struct, fullconcnt

def trainModel(cnn_struct, full_con_cnt, m_num):
	"""cnn_struct:((filterscnt,drop,pooling2x2(1-use,0-don't use)),...)"""
	logStruct(cnn_struct, full_con_cnt, m_num)
	
	inp_shape=(3, cls_win_height, cls_win_width)
	k = 0
	
	model = Sequential() 
	for layer in cnn_struct:
		f_cnt = layer[0]
		drop = layer[1]
		pooling = layer[2]
		
		if k == 0:	
			model.add(Convolution2D(f_cnt, (3, 3), activation='relu', input_shape=inp_shape))
			print("add input conv")
		else:
			model.add(Convolution2D(f_cnt, (3, 3), activation='relu'))
			print("add conv")
			
		if drop > 0:
			model.add(Dropout(drop))
			print("add dropout")
		if pooling == 1:
			model.add(MaxPooling2D(pool_size=(2, 2)))
			print("add pooling")
		
		k = k+1
	
	model.add(Flatten()) 
	model.add(Dense(full_con_cnt)) 
	model.add(Activation('relu')) 
	model.add(Dropout(0.3)) 
	model.add(Dense(1)) 
	model.add(Activation('sigmoid'))	
	
	model.compile(
		#loss='binary_crossentropy', 
		loss='mean_squared_error', 
		optimizer='rmsprop', 
		#optimizer='adagrad',
		metrics=['accuracy']) 
	
	train_datagen = ImageDataGenerator( 
		rescale=1. / 255,
		shear_range=0.2, 
		#zoom_range=[1,1.1],
		height_shift_range=0.05,
		width_shift_range=0.05,
		horizontal_flip=True
		)	
	
	
	test_datagen = ImageDataGenerator(
		rescale=1. / 255,
		horizontal_flip=True
		) 
	
	train_generator = train_datagen.flow_from_directory(
		train_data_dir,
		target_size=(cls_win_height, cls_win_width),
		batch_size=batch_size,
		class_mode='binary',
		color_mode='rgb'
		#save_to_dir=out_data_dir
		) 
	
	validation_generator = test_datagen.flow_from_directory(
		validation_data_dir,
		target_size=(cls_win_height, cls_win_width),
		batch_size=batch_size, class_mode='binary', color_mode='rgb') 
	
	cb1 = callbacks.CSVLogger(root_data_dir+"/res/log{number:05}.txt".format(number=m_num))
	
	
	model.fit_generator(
		train_generator,
		validation_data=validation_generator,
		steps_per_epoch=(int)(nb_train_samples/batch_size)*10,
		validation_steps=(int)(nb_validation_samples/batch_size)*10,
		epochs=nb_epochs,
		callbacks=[cb1]
		)	
	
	model.save(root_data_dir+'/res/nn{number:05}.h5'.format(number=m_num))
	#plot_model(model, to_file=root_data_dir+'/res/vis{number:05}.png'.format(number=m_num))
	
	
random.seed()
root_data_dir = ''
if len(sys.argv) > 1:
	root_data_dir = sys.argv[1]
else:
	print("use train.py <dataset_dir>")
	exit(1)
	
train_data_dir = root_data_dir + '/train/' 
validation_data_dir = root_data_dir + '/validation/' 
out_data_dir = root_data_dir + '/out/'

loadDataSetParams(root_data_dir+'/ds.conf')
print("w={}, h={}".format(cls_win_width, cls_win_height))

nb_train_samples = getSamplesCount(train_data_dir+'pos/')+getSamplesCount(train_data_dir+'neg/')
print("{} train samples".format(nb_train_samples))
nb_validation_samples = getSamplesCount(validation_data_dir+'pos/')+getSamplesCount(validation_data_dir+'neg/')
print("{} vld samples".format(nb_validation_samples))
nb_epochs = 50 
batch_size = 2 

m_num = 0
while True:
	struct = []
	#struct, fullconcnt = randomInit()
	
	#conv layers
	struct.append((5,0.3,1))
	struct.append((7,0.3,1))
	struct.append((9,0.3,0))	
			
	fullconcnt = 32
	
	print("=============================")
	print("cnn {}".format(m_num))
	
	trainModel(struct, fullconcnt, m_num)
	m_num = m_num+1
	break