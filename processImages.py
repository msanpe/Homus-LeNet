'''
Author: Miguel Sancho

Data augmenting script for the Homus dataset
'''


from __future__ import print_function
from PIL import Image, ImageOps
import numpy as np
import glob

import matplotlib.pyplot as plt

batch_size = 16
nb_classes = 32
nb_epoch = 30

# HOMUS contains images of 40 x 40 pixels
# input image dimensions for train
img_rows, img_cols = 40, 40

total = 0

#
# Load data from data/HOMUS/train_0, data/HOMUS/train_1,...,data/HOMUS_31 folders from HOMUS images
#
def load_data():
	global total
	image_list = []
	class_list = []
	for current_class_number in range(0,nb_classes):	# Number of class
		i = 0
		for filename in glob.glob('./data/HOMUS/train_{}/*.jpg'.format(current_class_number)):
			im = Image.open(filename).resize((img_rows,img_cols)).convert('L')
			im = ImageOps.invert(im)	# Meaning of grey level is 255 (black) and 0 (white)
			#im.show()
			rotR = im.rotate(-20)
			rotR = ImageOps.invert(rotR)
			rotR.save('./data/HOMUS/train_{}/{}.jpg'.format(current_class_number, i))
			i += 1
			rotL = im.rotate(20)
			rotL = ImageOps.invert(rotL)
			rotL.save('./data/HOMUS/train_{}/{}.jpg'.format(current_class_number, i))
			i += 1
			total += 1



load_data()
print(total)
