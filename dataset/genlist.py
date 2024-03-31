import os 

dataset = 'DR'

if dataset == 'HAM10000':
	domains = ['back', 'face', 'lowerExtremity', 'upperExtremity']
elif dataset == 'DR':
	domains = ['APTOS-2019', 'DDR', 'IDRiD']
else:
	raise KeyError('No such dataset exists!')

for domain in domains:
	log = open(dataset+'/'+domain+'_list.txt','w')
	directory = os.path.join(dataset, os.path.join(domain,'images'))
	classes = [x[0] for x in os.walk(directory)]
	classes = classes[1:]
	classes.sort()
	for idx,f in enumerate(classes):
		files = os.listdir(f)
		for file in files:
			s = os.path.abspath(os.path.join(f,file)) + ' ' + str(idx) + '\n'
			log.write(s)
	log.close()