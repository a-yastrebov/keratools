##################################################################
# Train new classifier in cascade
# Example: cls_train.py <index_of_new_classifier_in_cascade_from_1>
# 
# Copyright (c) 2017 Alexey Yastrebov
# MIT License, see LICENSE file.
##################################################################

from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Convolution2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K
import sys
import os

def getSamplesCount(path):
	cnt = 0
	filenames = os.listdir(path)
	for fn in filenames:
		if fn.endswith(".png") or fn.endswith(".bmp") or fn.endswith(".jpg"):
			cnt = cnt + 1
	return cnt

# dimensions of our images. 
cls_win_width, cls_win_height = 32, 60 

cc_num = 1
if len(sys.argv) > 1:
	cc_num = sys.argv[1]
	
root_data_dir = 'data/32x60cc{}/'.format(cc_num)
train_data_dir = root_data_dir + 'train/' 
validation_data_dir = root_data_dir + 'validation/'

nb_train_samples = getSamplesCount(train_data_dir+'pos/')+getSamplesCount(train_data_dir+'neg/')
print("{} train samples".format(nb_train_samples))
nb_validation_samples = getSamplesCount(validation_data_dir+'pos/')+getSamplesCount(validation_data_dir+'neg/')
print("{} vld samples".format(nb_validation_samples))
 
#nb_train_samples = 4563
#nb_validation_samples = 1093
nb_epochs = 10 
batch_size = 16
colorimage = False

if colorimage == True:
	inpshape=(3, cls_win_height, cls_win_width)
else:
	inpshape=(1, cls_win_height, cls_win_width)

model = Sequential() 
model.add(Convolution2D(16, (5, 5), activation='relu', input_shape=inpshape))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Flatten()) 
model.add(Dense(32)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1)) 
model.add(Activation('sigmoid')) 


model.compile(loss='binary_crossentropy', 
	optimizer='rmsprop', 
	metrics=['accuracy']) 


# this is the augmentation configuration we will use for training 
train_datagen = ImageDataGenerator( 
	rescale=1. / 255, 
	shear_range=0.2, 
	zoom_range=0.2 
	#horizontal_flip=True
	) 


# this is the augmentation configuration we will use for testing: 
# only rescaling 
test_datagen = ImageDataGenerator(
	rescale=1. / 255
	) 

if colorimage == True:
	colmode='rgb'
else:
	colmode='grayscale'	
	
train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(cls_win_height, cls_win_width),
	batch_size=batch_size,
	class_mode='binary',
	color_mode=colmode) 

validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(cls_win_height, cls_win_width), batch_size=batch_size, class_mode='binary', color_mode=colmode) 

model.fit_generator(
	train_generator,
	validation_data=validation_generator,
	steps_per_epoch=(int)(nb_train_samples/batch_size),
	validation_steps=(int)(nb_validation_samples/batch_size),
	epochs=nb_epochs
	)
 
model.save('nn{}.h5'.format(cc_num)) 
