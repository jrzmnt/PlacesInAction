import urllib2
import numpy as np
import sys
import progressbar
from PIL import Image
from cv2 import resize
from keras.models import Model
from vgg16_places_365 import VGG16_Places365
from os.path import join, realpath, dirname, exists, basename, splitext

def count_lines(f):

	lines = 0
	with open(f) as fi:
		for l in fi:
			lines += 1

	return lines

def write_features(feat_file, path, label, feat):

	feat_file.write(path)
	feat_file.write(' ')

	for i in feat:
		for j in i:
			feat_file.write('%s' % round(j,3))
			feat_file.write(' ')

	feat_file.write(label)
	feat_file.write('\n')

# Paths to the data
DOG_TRAIN = 'dog/train.txt'
DOG_VAL = 'dog/validation.txt'
DOG_TEST = 'dog/test.txt'

UCF_TRAIN = 'ucf/train.txt'
UCF_VAL = 'ucf/validation.txt'
UCF_TEST = 'ucf/test.txt'

PATHS = [DOG_TRAIN]

# Initializing the model
model = VGG16_Places365(weights='places', include_top=True)
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('fc2').output)

for path in PATHS:

	pb = progressbar.ProgressBar(count_lines(path))

	print 'Saving.../usr/share/datasets/CIARP/'+splitext(basename(path))[0]+'_dog_fc2.txt'
	print 'Saving.../usr/share/datasets/CIARP/'+splitext(basename(path))[0]+'_dog_softmax.txt'

	file_fc2 = open('/usr/share/datasets/CIARP/'+splitext(basename(path))[0]+'_dog_fc2.txt', 'w')
	file_softmax = open('/usr/share/datasets/CIARP/'+splitext(basename(path))[0]+'_dog_softmax.txt', 'w')

	with open(path) as f:
		for line in f:

			img_path = line.split()[0]
			label = line.split()[1]
			image = Image.open(img_path)
			image = np.array(image, dtype=np.uint8)
			image = resize(image, (224, 224))
			image = np.expand_dims(image, 0)		

			fc2_output = intermediate_layer_model.predict(image)
			softmax_output = model.predict(image)

			write_features(file_fc2, img_path, label, fc2_output)
			write_features(file_softmax, img_path, label, softmax_output)

			pb.update()