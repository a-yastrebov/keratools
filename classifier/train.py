##################################################################
# Train new classifier
# Example: train.py <dataset_dir>
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
cls_win_width, cls_win_height = 32*2+3, 60 

root_data_dir = ''
if len(sys.argv) > 1:
	root_data_dir = sys.argv[1]
else:
	print("use train.py <dataset_dir>")
	exit(1)
	
train_data_dir = root_data_dir + '/train/' 
validation_data_dir = root_data_dir + '/validation/' 

nb_train_samples = getSamplesCount(train_data_dir+'pos/')+getSamplesCount(train_data_dir+'neg/')
print("{} train samples".format(nb_train_samples))
nb_validation_samples = getSamplesCount(validation_data_dir+'pos/')+getSamplesCount(validation_data_dir+'neg/')
print("{} vld samples".format(nb_validation_samples))
nb_epochs = 20 
batch_size = 32 

model = Sequential() 
model.add(Convolution2D(2, (3, 3), activation='relu', input_shape=(3, cls_win_height, cls_win_width)))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(4, (3, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Convolution2D(8, (3, 3), activation='relu'))
model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Flatten()) 
model.add(Dense(16)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1)) 
model.add(Activation('sigmoid')) 


model.compile(loss='binary_crossentropy', 
	optimizer='rmsprop', 
	metrics=['accuracy']) 


train_datagen = ImageDataGenerator( 
	rescale=1. / 255
	#shear_range=0.2, 
	#zoom_range=0.2 
	#horizontal_flip=True
	) 

test_datagen = ImageDataGenerator(
	rescale=1. / 255
	) 

train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(cls_win_height, cls_win_width),
	batch_size=batch_size,
	class_mode='binary',
	color_mode='rgb') 

validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,
	target_size=(cls_win_height, cls_win_width),
	batch_size=batch_size, class_mode='binary', color_mode='rgb') 

cb1 = callbacks.CSVLogger("log1.txt")

model.fit_generator(
	train_generator,
	validation_data=validation_generator,
	steps_per_epoch=(int)(nb_train_samples/batch_size),
	validation_steps=(int)(nb_validation_samples/batch_size),
	epochs=nb_epochs,
	callbacks=[cb1]
	)
 
model.save('nn.h5') 
