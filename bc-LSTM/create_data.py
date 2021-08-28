import numpy as np, pandas as pd
from collections import defaultdict
import pickle
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()




pre_data = np.asarray(pd.read_csv("./data/transcripts.csv" , header=None))

train = pd.read_csv("./data/text_train.csv", header=None)
test = pd.read_csv("./data/text_test.csv", header=None)
train = np.asarray(train)
test = np.asarray(test)
train_index = np.asarray(train[:,0], dtype = 'int')
test_index = np.asarray(test[:,0], dtype = 'int')



def main(name):

	path = "./data/"+name+"/"+name
	print path
	train_video_mapping=defaultdict(list)
	train_video_mapping_index=defaultdict(list)
	test_video_mapping=defaultdict(list)
	test_video_mapping_index=defaultdict(list)

	data_train = np.asarray(pd.read_csv(path+"_train0.csv", header=None))
	data_test = np.asarray(pd.read_csv(path+"_test0.csv", header=None))

	for i in xrange(train_index.shape[0]):
		train_video_mapping[pre_data[train_index[i]][0].rsplit("_",1)[0] ].append(train_index[i])
		train_video_mapping_index[pre_data[train_index[i]][0].rsplit("_",1)[0] ].append( int(pre_data[train_index[i]][0].rsplit("_",1)[1]) )

	for i in xrange(test_index.shape[0]):
		test_video_mapping[pre_data[test_index[i]][0].rsplit("_",1)[0] ].append(test_index[i])
		test_video_mapping_index[pre_data[test_index[i]][0].rsplit("_",1)[0] ].append( int(pre_data[test_index[i]][0].rsplit("_",1)[1]) )

	train_indices = dict((c, i) for i, c in enumerate(train_index))
	test_indices = dict((c, i) for i, c in enumerate(test_index))

	max_len = 0
	for key,value in train_video_mapping.iteritems():
		max_len = max(max_len , len(value))
	for key,value in test_video_mapping.iteritems():
		max_len = max(max_len, len(value))

	pad = np.asarray([0 for i in xrange(data_train[0][:-1].shape[0])])

	print "Mapping train"

	train_data_X =[]
	train_data_Y =[]
	train_length =[]
	for key,value in train_video_mapping.iteritems():

		
		lst = np.column_stack((train_video_mapping_index[key],value)  )
		ind = np.asarray(sorted(lst,key=lambda x: x[0]))


		lst_X, lst_Y=[],[]
		ctr=0;
		for i in xrange(ind.shape[0]):
			ctr+=1
			#lst_X.append(preprocessing.scale( min_max_scaler.fit_transform(data_train[train_indices[ind[i,1]]][:-1])))
			lst_X.append(data_train[train_indices[ind[i,1]]][:-1])
			lst_Y.append(data_train[train_indices[ind[i,1]]][-1])
		train_length.append(ctr)
		for i in xrange(ctr, max_len):
			lst_X.append(pad)
			lst_Y.append(0) #dummy label
		
		train_data_X.append(lst_X)
		train_data_Y.append(lst_Y)
	

	test_data_X =[]
	test_data_Y =[]
	test_length =[]

	print "Mapping test"

	for key,value in test_video_mapping.iteritems():

		lst = np.column_stack((test_video_mapping_index[key],value)  )
		ind = np.asarray(sorted(lst,key=lambda x: x[0]))

		lst_X, lst_Y=[],[]
		ctr=0
		for i in xrange(ind.shape[0]):
			ctr+=1
			#lst_X.append(preprocessing.scale( min_max_scaler.transform(data_test[test_indices[ind[i,1]]][:-1])))
			lst_X.append(data_test[test_indices[ind[i,1]]][:-1])
			lst_Y.append(data_test[test_indices[ind[i,1]]][-1])
		test_length.append(ctr)
		for i in xrange(ctr, max_len):
			lst_X.append(pad)
			lst_Y.append(0) #dummy label

		test_data_X.append(np.asarray(lst_X))
		test_data_Y.append(np.asarray(lst_Y))

	train_data_X = np.asarray(train_data_X)
	test_data_X = np.asarray(test_data_X)
	print train_data_X.shape, test_data_X.shape,len(train_length), len(test_length)

	print "Dumping data"
	with open('./input/'+name+'.pickle', 'wb') as handle:
		pickle.dump((train_data_X,  np.asarray(train_data_Y), test_data_X, np.asarray(test_data_Y), max_len ,train_length, test_length), handle, protocol=pickle.HIGHEST_PROTOCOL)



	


if __name__ == "__main__":

	names = ['text','audio','video']
	for nm in names:
		main(nm)
