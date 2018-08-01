path_file = 'train_ex.txt'

path_write = open('train_ex_adjusted.txt','w')

with open(path_file) as f:
	for line in f:
		label = line.split()[-1]
		path = line.split()[0]
		feat = line.split()[1:-2]

		path_write.write(path+' '+label+' '+)
