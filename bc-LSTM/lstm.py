import numpy as np
from keras.layers import Input, LSTM, Dense, TimeDistributed, Masking, Dropout, Bidirectional
from keras.models import Model
from keras import backend as K
import theano.tensor as T
import theano
import pickle
import sys
import argparse
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
np.random.seed(1337) # for reproducibility

unimodal_activations={}

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def createOneHot(train_label,  test_label):


	maxlen = int(max(train_label.max(), test_label.max()))
	
	train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen+1))
	test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen+1))
	
	for i in xrange(train_label.shape[0]):
		for j in xrange(train_label.shape[1]):
			train[i,j,train_label[i,j]]=1

	for i in xrange(test_label.shape[0]):
		for j in xrange(test_label.shape[1]):
			test[i,j,test_label[i,j]]=1

	return train,  test

def createVal(train_data, train_mask, train_label, valid_portion=None):

	n_samples = train_data.shape[0]
	sidx = np.arange(n_samples)
	n_train = int(np.round(n_samples * (1. - valid_portion)))

	val_data = np.asarray([train_data[s] for s in sidx[n_train:]])
	val_mask = np.asarray([train_mask[s] for s in sidx[n_train:]])
	val_label = np.asarray([train_label[s] for s in sidx[n_train:]])

	train_data = np.asarray([train_data[s] for s in sidx[:n_train]])
	train_mask = np.asarray([train_mask[s] for s in sidx[:n_train]])
	train_label = np.asarray([train_label[s] for s in sidx[:n_train]])

	return train_data, train_mask, train_label, val_data, val_mask, val_label


def calc_test_result(result, test_label, test_mask):

	true_label=[]
	predicted_label=[]

	for i in xrange(result.shape[0]):
		for j in xrange(result.shape[1]):
			if test_mask[i,j]==1:
				true_label.append(np.argmax(test_label[i,j] ))
				predicted_label.append(np.argmax(result[i,j] ))
		
	print "Confusion Matrix :"
	print confusion_matrix(true_label, predicted_label)
	print "Classification Report :"
	print classification_report(true_label, predicted_label)
	print "Accuracy ", accuracy_score(true_label, predicted_label)

def unimodal(mode):

	print 'starting unimodal ', mode
	with open('./input/'+mode+'.pickle', 'rb') as handle:
			(train_data, train_label, test_data, test_label, maxlen, train_length, test_length) = pickle.load(handle)

	train_label = train_label.astype('int')
	test_label = test_label.astype('int')

	train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
	for i in xrange(len(train_length)):
		train_mask[i,:train_length[i]]=1.0

	test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
	for i in xrange(len(test_length)):
		test_mask[i,:test_length[i]]=1.0

	train_label, test_label = createOneHot(train_label, test_label)



	input_data = Input(shape=(train_data.shape[1],train_data.shape[2]))
	masked = Masking(mask_value =0)(input_data)
	lstm = Bidirectional(LSTM(300, activation='tanh', return_sequences = True, dropout=0.6))(masked)
	inter = Dropout(0.9)(lstm)
	inter1 = TimeDistributed(Dense(100,activation='tanh'))(inter)
	inter = Dropout(0.9)(inter1)
	output = TimeDistributed(Dense(2,activation='softmax'))(inter)

	model = Model(input_data, output)
	aux = Model(input_data, inter1)
	model.compile(optimizer='adadelta', loss='categorical_crossentropy', sample_weight_mode='temporal')
	early_stopping = EarlyStopping(monitor='val_loss', patience=10)
	model.fit(train_data, train_label,
	                epochs=200,
	                batch_size=10,
	                sample_weight=train_mask,
	                shuffle=True, 
	                callbacks=[early_stopping],
	                validation_split=0.2)
	                

	model.save('./models/'+mode+'.h5') 

	train_activations = aux.predict(train_data)
	test_activations = aux.predict(test_data)
	
	unimodal_activations[mode+'_train']=train_activations
	unimodal_activations[mode+'_test']=test_activations

	unimodal_activations['train_mask']=train_mask
	unimodal_activations['test_mask']= test_mask
	unimodal_activations['train_label']=train_label
	unimodal_activations['test_label']=test_label



def multimodal(unimodal_activations):

	print "starting multimodal"
	#Fusion (appending) of features

	train_data = np.concatenate((unimodal_activations['text_train'], unimodal_activations['audio_train'], unimodal_activations['video_train']), axis=2)
	test_data = np.concatenate((unimodal_activations['text_test'], unimodal_activations['audio_test'], unimodal_activations['video_test']), axis=2)
	train_mask=unimodal_activations['train_mask']
	test_mask=unimodal_activations['test_mask']
	train_label=unimodal_activations['train_label']
	test_label=unimodal_activations['test_label']

	#Multimodal model

	input_data = Input(shape=(train_data.shape[1],train_data.shape[2]))
	masked = Masking(mask_value =0)(input_data)
	lstm = Bidirectional(LSTM(300, activation='tanh', return_sequences = True, dropout=0.4))(masked)
	inter = Dropout(0.9)(lstm)
	inter1 = TimeDistributed(Dense(500,activation='relu'))(inter)
	inter = Dropout(0.9)(inter1)
	output = TimeDistributed(Dense(2,activation='softmax'))(inter)

	model = Model(input_data, output)
	aux = Model(input_data, inter1)
	model.compile(optimizer='adadelta', loss='categorical_crossentropy', sample_weight_mode='temporal')
	early_stopping = EarlyStopping(monitor='val_loss', patience=10)
	model.fit(train_data, train_label,
	                epochs=200,
	                batch_size=10,
	                sample_weight=train_mask,
	                shuffle=True, 
	                callbacks=[early_stopping],
	                validation_split=0.2)
	                

	model.save('./models/multimodal.h5') 

	result = model.predict(test_data)
	calc_test_result(result, test_label, test_mask)	          


if __name__=="__main__":
	
	argv = sys.argv[1:]
	parser = argparse.ArgumentParser()
	parser.add_argument("--unimodal", type=str2bool, nargs='?',
	                    const=True, default=False)
	args, _ = parser.parse_known_args(argv)

	if args.unimodal:

		print "Training unimodals first"

		modality = ['text', 'audio', 'video']
		for mode in modality:
			unimodal(mode)

		print "Saving unimodal activations"
		with open('unimodal.pickle', 'wb') as handle:
			pickle.dump(unimodal_activations, handle, protocol=pickle.HIGHEST_PROTOCOL)


	with open('unimodal.pickle', 'rb') as handle:
	    unimodal_activations = pickle.load(handle)


	multimodal(unimodal_activations)
