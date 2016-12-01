/'''
Trains a simple LeNet-5 (http://yann.lecun.com/exdb/lenet/) adapted to the HOMUS dataset using Keras Software (http://keras.io/)

LeNet-5 demo example http://eblearn.sourceforge.net/beginner_tutorial2_train.html

This example executed with 8x8 reescaled images and 50 epochs obtains an accuracy close to 32%.
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
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import TensorBoard
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

batch_size = 16
nb_classes = 32
nb_epoch = 20

# HOMUS contains images of 40 x 40 pixels
# input image dimensions for training
img_rows, img_cols = 40, 40

# number of convolutional filters to use
nb_filters1 = 6
nb_filters2 = 64
nb_filters3 = 120

# convolution kernel size
nb_conv1 = 5
nb_conv2 = 6
nb_conv3 = 1

# size of pooling area for max pooling
nb_pool = 2

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

	Y = np.asarray(class_list)

	return X, Y, input_shape

X, Y, input_shape = load_data()

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

	# set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(500))
	model.add(Activation("relu"))

	# softmax classifier
	model.add(Dense(nb_classes))
	model.add(Activation("softmax"))

	return model

seed = 8
np.random.seed(seed);
optimizer = SGD(lr=0.01)

kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=seed)
cvscores = []
i = 0

for train, test in kfold.split(X, Y):
	print ('fold {}'.format(i +1))
	model = create_model()
	model.compile(loss = 'categorical_crossentropy',optimizer = optimizer, metrics = ['accuracy'])
	yTrain = np_utils.to_categorical(Y[train], nb_classes)
	yTest = np_utils.to_categorical(Y[test], nb_classes)
	model.fit(X[train], yTrain, nb_epoch=nb_epoch, batch_size=10, verbose=2)
	# evaluate the model
	scores = model.evaluate(X[test], yTest, verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
	++i
#
# Results
#
f = open('workfile', 'w')
for score in cvscores:
	f.write('{}\n'.format(score))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
#print('Test score:', score[0])
#print('Test accuracy:', score[1])


# file name to save model
filename = 'homus_cnn.h5'

# save network model
model.save(filename)

# load neetwork model
#model = load_model(filename)
