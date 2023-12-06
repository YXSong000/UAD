import os 

dataset = 'DR'

if dataset == 'office':
	domains = ['amazon', 'dslr', 'webcam']
elif dataset == 'office-caltech':
	domains = ['amazon', 'dslr', 'webcam', 'caltech']
elif dataset == 'office-home':
	domains = ['Art', 'Clipart', 'Product', 'Real_World']
elif dataset == 'HAM10000':
	domains = ['abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot', 'genital', 'hand', 'lowerExtremity', 'neck', 'scalp', 'trunk', 'unknown', 'upperExtremity']
elif dataset == 'DR':
	domains = ['APTOS-2019', 'DDR', 'IDRiD', 'Messidor-2']
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