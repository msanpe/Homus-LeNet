'''
Trains a simple LeNet-5 (http://yann.lecun.com/exdb/lenet/) adapted to the HOMUS dataset using Keras Software (http://keras.io/)
LeNet-5 demo example http://eblearn.sourceforge.net/beginner_tutorial2_train.html

This example executed with 40x40 reescaled images and 30 epochs obtains an accuracy close to 95%.
'''

from __future__ import print_function
from PIL import Image, ImageOps
import numpy as np
import glob

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, adam, adadelta
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras import backend as K
import matplotlib.pyplot as plt

batch_size = 16
nb_classes = 32
nb_epoch = 30

# HOMUS contains images of 40 x 40 pixels
# input image dimensions for training
img_rows, img_cols = 40, 40

#
# Load data from data/HOMUS/train_0, data/HOMUS/train_1,...,data/HOMUS_31 folders from HOMUS images
#
def load_data():
	image_list = []
	class_list = []
	for current_class_number in range(0,nb_classes):	# Number of class
		for filename in glob.glob('./data/HOMUS/train_{}/*.jpg'.format(current_class_number)):
			im = Image.open(filename).resize((img_rows,img_cols)).convert('L')
			im = ImageOps.invert(im)	# Meaning of grey level is 255 (black) and 0 (white)
			#im.show()
			image_list.append(np.asarray(im).astype('float32')/255)
			class_list.append(current_class_number)

	n = len(image_list)	# Total examples

	if K.image_dim_ordering() == 'th':
		X = np.asarray(image_list).reshape(n,1,img_rows,img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		X = np.asarray(image_list).reshape(n,img_rows,img_cols,1)
		input_shape = (img_rows, img_cols, 1)

	Y = np_utils.to_categorical(np.asarray(class_list), nb_classes)

	# Shuffle (X,Y)
	randomize = np.arange(len(Y))
	np.random.shuffle(randomize)
	X, Y = X[randomize], Y[randomize]

	n_partition = int(n*0.9)	# Train 90% and Test 10%

	X_train = X[:n_partition]
	Y_train = Y[:n_partition]

	X_test  = X[n_partition:]
	Y_test  = Y[n_partition:]

	return X_train, Y_train, X_test, Y_test, input_shape

# the data split between train and test sets
X_train, Y_train, X_test, Y_test, input_shape = load_data()

print(X_train.shape, 'train samples')
print(X_test.shape, 'test samples')
print(input_shape,'input_shape')
print(nb_epoch,'epochs')

#
# Neural Network Structure
#
def create_model():
	model = Sequential()
	# first set of CONV => RELU => POOL
	model.add(Convolution2D(20, 5, 5, border_mode="same",
	input_shape=(input_shape)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# second set of CONV => RELU => POOL
	model.add(Convolution2D(50, 5, 5, border_mode="same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.5)) # reducing overfitting

	# set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(500))
	model.add(Activation("relu"))

	# softmax classifier
	model.add(Dense(nb_classes))
	model.add(Activation("softmax"))
	##model.summary()

	return model


optimizer = SGD(lr = 0.01,momentum=0.1,nesterov = False)
early_stopping = EarlyStopping(monitor='loss', patience=3)
#model = load_model('homus_cnn.h5')
model = create_model()
model.compile(loss = 'categorical_crossentropy',optimizer = optimizer, metrics = ['accuracy'])

history = model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch = nb_epoch,
	verbose = 2, validation_data = (X_test, Y_test), callbacks=[early_stopping])
score = model.evaluate(X_test, Y_test, verbose = 1)

#
# Results
#

print('Test score:', score[0])
print('Test accuracy:', score[1])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# file name to save model
filename = 'homus_cnn.h5'

# save network model
#model.save(filename)

# load neetwork model
#model = load_model(filename)
