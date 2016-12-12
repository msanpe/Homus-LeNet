from __future__ import print_function
from PIL import Image, ImageOps
import numpy as np
import glob

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import timeit

batch_size = 16
nb_classes = 32

img_rows, img_cols = 40, 40


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


	X = np.asarray(image_list).reshape(n, -1)
	Y = np.asarray(class_list)

	n_partition = int(n*0.9)	# Train 90% and Test 10%

	X_train = X[:n_partition]
	Y_train = Y[:n_partition]

	X_test  = X[n_partition:]
	Y_test  = Y[n_partition:]

	return X_train, Y_train, X_test, Y_test, n

def run_svms():
	X_train, Y_train, X_test, Y_test, n = load_data()
	accuracies = []
	print ("\n\nTraining SVM with data set size {}".format(n))
	print(X_train.shape)
	print(Y_train.shape)
	print(Y_test[1])
	clf = svm.SVC(gamma=0.001)
	clf.fit(X_train, Y_train)
	predictions = [int(a) for a in clf.predict(X_test)]

	pruebaP = clf.predict(X_test[1])
	print('predicho')
	print(pruebaP)
	print('Actual')

	accuracy = sum(int(a == y) for a, y in zip(predictions, Y_test)) / 100.0
	print ("Accuracy was {} percent".format(accuracy))
	accuracies.append(accuracy)

	f = open("more_data_svm.json", "w")
	json.dump(accuracies, f)
	f.close()

run_svms()
