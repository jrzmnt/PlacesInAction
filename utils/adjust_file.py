import progressbar

path_file = '/usr/share/datasets/CIARP/dog_places/train_dog_softmax.txt'
path_write = open('/usr/share/datasets/CIARP/dog_places/train_dog_softmax_adjusted.txt','w')



def count_lines(f):

	lines = 0
	with open(f) as fi:
		for l in fi:
			lines += 1

	return lines

pb = progressbar.ProgressBar(count_lines(path_file))

with open(path_file) as f:

	for line in f:
		label = line.split()[-1]
		path = line.split()[0]
		feat = line.split()[1:-2]

		path_write.write(path+' '+label)

		for f in feat:
			path_write.write(' ')
			path_write.write(f)

		path_write.write('\n')
		pb.update()