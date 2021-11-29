import os
import sys
import re
import json
import pickle

import h5py
import nltk
import numpy as np
import jsonlines
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

import config


def pickle_loader(filename):
    if sys.version_info[0] < 3:
        return pickle.load(open(filename, 'rb'))
    else:
        return pickle.load(open(filename, 'rb'), encoding="latin1")


class DataLoader:

    DATA_PATH_JSON = "./data/sarcasm_data.json"
    AUDIO_PICKLE = "./data/audio_features.p"
    INDICES_FILE = "./data/split_indices.p"
    GLOVE_DICT = "./data/glove_full_dict.p"
    BERT_TARGET_EMBEDDINGS = "./data/bert-output.jsonl"
    BERT_CONTEXT_EMBEDDINGS = "./data/bert-output-context.jsonl"
    UTT_ID = 0
    CONTEXT_ID = 2
    SHOW_ID = 9
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    def __init__(self, config):

        self.config = config
        
        dataset_json = json.load(open(self.DATA_PATH_JSON))

        if config.use_bert and config.use_target_text:
            text_bert_embeddings = []
            with jsonlines.open(self.BERT_TARGET_EMBEDDINGS) as reader:
                
                # Visit each target utterance
                for obj in reader:

                    CLS_TOKEN_INDEX = 0
                    features = obj['features'][CLS_TOKEN_INDEX]

                    bert_embedding_target = []
                    for layer in [0,1,2,3]:
                        bert_embedding_target.append(np.array(features["layers"][layer]["values"]))
                    bert_embedding_target = np.mean(bert_embedding_target, axis=0)
                    text_bert_embeddings.append(np.copy(bert_embedding_target))
        else:
            text_bert_embeddings = None


        if config.use_context:
            context_bert_embeddings = self.loadContextBert(dataset_json)
        else:
            context_bert_embeddings = None



        if config.use_target_audio:
            audio_features = pickle_loader(self.AUDIO_PICKLE)
        else:
            audio_features = None

        if config.use_target_video:
            video_features_file = h5py.File('data/features/utterances_final/resnet_pool5.hdf5')
            context_video_features_file = h5py.File('data/features/context_final/resnet_pool5.hdf5')
        else:
            video_features_file = None
            context_video_features_file = None


        self.parseData(dataset_json, audio_features, video_features_file, context_video_features_file, text_bert_embeddings, context_bert_embeddings)

        if config.use_target_video:
            video_features_file.close()
            context_video_features_file.close()

        self.StratifiedKFold()
        self.setupGloveDict()

        # Setup speaker independent split
        self.speakerIndependentSplit()


    def parseData(self, json, audio_features, video_features_file=None, context_video_features_file=None, text_bert_embeddings=None, context_bert_embeddings=None):
        '''
        Prepares json data into lists
        data_input = [ (utterance:string, speaker:string, context:list_of_strings, context_speakers:list_of_strings, utterance_audio:features ) ]
        data_output = [ sarcasm_tag:int ]
        '''
        self.data_input, self.data_output = [], []
        
        for idx, ID in enumerate(json.keys()):
            self.data_input.append((json[ID]["utterance"], json[ID]["speaker"], json[ID]["context"],
                                    json[ID]["context_speakers"], audio_features[ID] if audio_features else None,
                                    video_features_file[ID][()] if video_features_file else None,
                                    context_video_features_file[ID][()] if context_video_features_file else None,
                                    text_bert_embeddings[idx] if text_bert_embeddings else None,
                                    context_bert_embeddings[idx] if context_bert_embeddings else None,
                                    json[ID]["show"]))
            self.data_output.append( int(json[ID]["sarcasm"]) )

    def loadContextBert(self, dataset, ):

        # Prepare context video length list
        length=[]
        for idx, ID in enumerate(dataset.keys()):
            length.append(len(dataset[ID]["context"]))

        # Load BERT embeddings
        with jsonlines.open(self.BERT_CONTEXT_EMBEDDINGS) as reader:
            context_utterance_embeddings=[]
            # Visit each context utterance
            for obj in reader:

                CLS_TOKEN_INDEX = 0
                features = obj['features'][CLS_TOKEN_INDEX]

                bert_embedding_target = []
                for layer in [0,1,2,3]:
                    bert_embedding_target.append(np.array(features["layers"][layer]["values"]))
                bert_embedding_target = np.mean(bert_embedding_target, axis=0)
                context_utterance_embeddings.append(np.copy(bert_embedding_target))

        # Checking whether total context features == total context sentences
        assert(len(context_utterance_embeddings)== sum(length))

        # Rearrange context features for each target utterance
        cumulative_length = [length[0]]
        cumulative_value = length[0]
        for val in length[1:]:
            cumulative_value+=val
            cumulative_length.append(cumulative_value)

        assert(len(length)==len(cumulative_length))

        end_index = cumulative_length
        start_index = [0]+cumulative_length[:-1]

        final_context_bert_features = []
        for start, end in zip(start_index, end_index):
            local_features = []
            for idx in range(start, end):
                local_features.append(context_utterance_embeddings[idx])
            final_context_bert_features.append(local_features)

        return final_context_bert_features



    def StratifiedKFold(self, splits=5):
        '''
        Prepares or loads (if existing) splits for k-fold 
        '''
        skf = StratifiedKFold(n_splits=splits, shuffle=True)
        split_indices = [(train_index, test_index) for train_index, test_index in skf.split(self.data_input, self.data_output)]

        if not os.path.exists(self.INDICES_FILE):
            pickle.dump(split_indices, open(self.INDICES_FILE, 'wb'), protocol=2)
        

    def getStratifiedKFold(self):
        '''
        Returns train/test indices for k-folds
        '''
        self.split_indices = pickle_loader(self.INDICES_FILE)
        return self.split_indices

    def speakerIndependentSplit(self):
        '''
        Prepares split for speaker independent setting
        Train: Fr, TGG, Sa
        Test: TBBT
        '''
        self.train_ind_SI, self.test_ind_SI = [], []
        for ind, data in enumerate(self.data_input):
            if data[self.SHOW_ID] == "FRIENDS":
                self.test_ind_SI.append(ind)
            else:
                self.train_ind_SI.append(ind)

    def getSpeakerIndependent(self):
        '''
        Returns the split indices of speaker independent setting
        '''
        return self.train_ind_SI, self.test_ind_SI



    def getSplit(self, indices):
        '''
        Returns the split comprising of the indices
        '''
        data_input = [self.data_input[ind] for ind in indices]
        data_output = [self.data_output[ind] for ind in indices]
        return data_input, data_output



    def fullDatasetVocab(self):
        '''
        Return the full dataset's vocabulary to filter and cache glove embedding dictionary
        '''

        vocab = defaultdict(lambda:0)
        utterances = [instance[self.UTT_ID] for instance in self.data_input]
        contexts = [instance[self.CONTEXT_ID] for instance in self.data_input]


        for utterance in utterances:
            clean_utt = DataHelper.clean_str(utterance)
            utt_words = nltk.word_tokenize(clean_utt)
            for word in utt_words:
                vocab[word.lower()] += 1

        for context in contexts:
            for c_utt in context:
                clean_utt = DataHelper.clean_str(c_utt)
                utt_words = nltk.word_tokenize(clean_utt)
                for word in utt_words:
                    vocab[word.lower()] += 1
        return vocab


    def setupGloveDict(self):
        '''
        Caching the glove dictionary based on all the words in the dataset.
        This cache is later used to create appropriate dictionaries for each fold's training vocabulary
        '''
        assert(self.config.word_embedding_path is not None)

        # Vocabulary of the full dataset
        vocab = self.fullDatasetVocab()

        if os.path.exists(self.GLOVE_DICT):
            self.wordemb_dict = pickle_loader(self.GLOVE_DICT)
        else:   
            self.wordemb_dict = {}
            for line in open(self.config.word_embedding_path,'r'):
                splitLine = line.split() 
                word = splitLine[0]
                try:
                    embedding = np.array([float(val) for val in splitLine[1:]])

                    # Filter glove words based on its presence in the vocab
                    if word.lower() in vocab:
                        self.wordemb_dict[word.lower()] = embedding
                except:
                    print("Error word in glove file (skipped): ", word)
                    continue
            self.wordemb_dict[self.PAD_TOKEN] = np.zeros(self.config.embedding_dim)
            self.wordemb_dict[self.UNK_TOKEN] = np.random.uniform(-0.25,0.25,self.config.embedding_dim)

            pickle.dump(self.wordemb_dict, open(self.GLOVE_DICT, "wb"))
            













class DataHelper:

    UTT_ID = 0
    SPEAKER_ID = 1
    CONTEXT_ID = 2
    CONTEXT_SPEAKERS_ID = 3
    TARGET_AUDIO_ID = 4
    TARGET_VIDEO_ID = 5
    CONTEXT_VIDEO_ID = 6
    TEXT_BERT_ID = 7
    CONTEXT_BERT_ID = 8

    PAD_ID = 0
    UNK_ID = 1
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    GLOVE_MODELS = "./data/temp/glove_dict_{}.p"
    GLOVE_MODELS_CONTEXT = "./data/temp/glove_dict_context_{}.p"


    def __init__(self, train_input, train_output, test_input, test_output, config, dataLoader):
        self.dataLoader = dataLoader
        self.config = config
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output

        # create vocab for current split train set
        self.createVocab(config.use_context)
        print("vocab size: " + str(len(self.vocab)))


        self.loadGloveModelForCurrentSplit(config.use_context)
        self.createEmbeddingMatrix()


    @staticmethod
    def clean_str(string):
        '''
        Tokenization/string cleaning.
        '''
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
        string = re.sub(r"\'s", " \'s", string) 
        string = re.sub(r"\'ve", " \'ve", string) 
        string = re.sub(r"n\'t", " n\'t", string) 
        string = re.sub(r"\'re", " \'re", string) 
        string = re.sub(r"\'d", " \'d", string) 
        string = re.sub(r"\'ll", " \'ll", string) 
        string = re.sub(r",", " , ", string) 
        string = re.sub(r"!", " ! ", string) 
        string = re.sub(r"\"", " \" ", string) 
        string = re.sub(r"\(", " ( ", string) 
        string = re.sub(r"\)", " ) ", string) 
        string = re.sub(r"\?", " ? ", string) 
        string = re.sub(r"\s{2,}", " ", string) 
        string = re.sub(r"\.", " . ", string)    
        string = re.sub(r".\, ", " , ", string)  
        string = re.sub(r"\\n", " ", string)  
        return string.strip().lower()



    def getData(self, ID=None, mode=None, error_message=None):

        if mode == "train":
            return [instance[ID] for instance in self.train_input]
        elif mode == "test":
            return [instance[ID] for instance in self.test_input]
        else:
            print(error_message)
            exit()



    def createVocab(self, use_context=False):

        self.vocab = vocab = defaultdict(lambda:0)
        utterances = self.getData(self.UTT_ID, mode="train")

        for utterance in utterances:
            clean_utt = self.clean_str(utterance)
            utt_words = nltk.word_tokenize(clean_utt)
            for word in utt_words:
                vocab[word.lower()] += 1

        # Add vocabulary fron context sentences of train split if context is used
        if use_context:
            context_utterances = self.getData(self.CONTEXT_ID, mode="train")
            for context in context_utterances:
                for c_utt in context:
                    clean_utt = self.clean_str(c_utt)
                    utt_words = nltk.word_tokenize(clean_utt)
                    for word in utt_words:
                        vocab[word.lower()] += 1



    def loadGloveModelForCurrentSplit(self, use_context=False):
        '''
        Loads the Glove pre-trained model for the current split
        '''
        
        print("Loading glove model")

        # if model already exists:
        filename = self.GLOVE_MODELS_CONTEXT if use_context else self.GLOVE_MODELS
        filename = filename.format(self.config.fold)

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        if os.path.exists(filename):
            self.model = pickle_loader(filename)
            self.embed_dim = len(self.dataLoader.wordemb_dict[self.PAD_TOKEN])
        else:
            self.model = model = {}
            self.embed_dim = 0

            # Further filter glove dict words to contain only train set vocab for current fold
            for word, embedding in self.dataLoader.wordemb_dict.items():
                if word in self.vocab: model[word.lower()] = embedding
                self.embed_dim = len(embedding)

            pickle.dump(self.model, open(filename, "wb"), protocol=2)


    def createEmbeddingMatrix(self):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        also creates word_idx_map : to map all words to proper index of i for associated
        embedding matrix W
        """

        vocab_size = len(self.model) # length of filtered glove embedding words
        self.word_idx_map = word_idx_map = dict()
        self.W = W = np.zeros(shape=(vocab_size+2, self.embed_dim), dtype='float32')            
        
        # Pad and Unknown
        W[self.PAD_ID] = self.dataLoader.wordemb_dict[self.PAD_TOKEN]
        W[self.UNK_ID] = self.dataLoader.wordemb_dict[self.UNK_TOKEN]
        word_idx_map[self.PAD_TOKEN] = self.PAD_ID
        word_idx_map[self.UNK_TOKEN] = self.UNK_ID

        # Other words
        i = 2
        for word in self.model:
            if (word != self.PAD_TOKEN) and (word != self.UNK_TOKEN):
                W[i] = np.copy(self.model[word])
                word_idx_map[word] = i
                i += 1

        # Make words not in glove as unknown
        for word in self.vocab:
            if word not in self.model:
                word_idx_map[word] = self.UNK_ID


    def getEmbeddingMatrix(self):
        return self.W


    def wordToIndex(self, utterance):

        word_indices = [self.word_idx_map.get(word, self.UNK_ID) for word in nltk.word_tokenize(self.clean_str(utterance))]

        #padding to max_sent_length
        word_indices = word_indices[:self.config.max_sent_length]
        word_indices = word_indices + [self.PAD_ID]*(self.config.max_sent_length - len(word_indices))
        assert(len(word_indices) == self.config.max_sent_length)
        return word_indices


    def getTargetBertFeatures(self, mode=None):

        utterances = self.getData(self.TEXT_BERT_ID, mode, 
                                  "Set mode properly for vectorizeUtterance method() : mode = train/test")

        return utterances

    def getContextBertFeatures(self, mode=None):

        utterances = self.getData(self.CONTEXT_BERT_ID, mode, 
                                  "Set mode properly for vectorizeUtterance method() : mode = train/test")

        mean_features=[]
        for utt in utterances:
            mean_features.append(np.mean(utt, axis=0))

        return np.array(mean_features)


    def vectorizeUtterance(self, mode=None):

        
        utterances = self.getData(self.UTT_ID, mode, 
                                  "Set mode properly for vectorizeUtterance method() : mode = train/test")

        vector_utt = []
        for utterance in utterances:
            word_indices = self.wordToIndex(utterance)
            vector_utt.append(word_indices)

        return vector_utt


    def getAuthor(self, mode=None):

        authors = self.getData(self.SPEAKER_ID, mode, 
                               "Set mode properly for contextMask method() : mode = train/test")

        # Create dictionary for speaker

        if mode=="train":
            author_list = set()
            author_list.add("PERSON")

            for author in authors:
                author = author.strip()
                if "PERSON" not in author:
                    author_list.add(author)

            self.author_ind={author:ind for ind, author in enumerate(author_list)}
            self.UNK_AUTHOR_ID = self.author_ind["PERSON"]
            self.config.num_authors = len(self.author_ind)
        
        # Convert authors into author_ids
        authors = [self.author_ind.get(author.strip(), self.UNK_AUTHOR_ID) for author in authors]
        authors = self.toOneHot(authors, len(self.author_ind))
        return authors
        

    def vectorizeContext(self, mode=None):

        dummy_sent = [self.PAD_ID]*self.config.max_sent_length

        contexts = self.getData(self.CONTEXT_ID, mode, 
                                "Set mode properly for vectorizeContext method() : mode = train/test")

        vector_context = []
        for context in contexts:
            local_context = []
            for utterance in context[-self.config.max_context_length:]: # taking latest (max_context_length) sentences
                #padding to max_sent_length
                word_indices = self.wordToIndex(utterance)
                local_context.append(word_indices)
            for _ in range(self.config.max_context_length - len(local_context)):
                local_context.append(dummy_sent[:])
            local_context = np.array(local_context)
            vector_context.append(local_context)

        return np.array(vector_context)


    def pool_text(self, data):

        data_vector = [self.W[ind] for ind in data if ind != 0] # only pick up non pad words
        data_vector = np.mean(data_vector, axis=0)
        return data_vector

    def getContextPool(self, mode=None):

        contexts = self.getData(self.CONTEXT_ID, mode, 
                                "Set mode properly for vectorizeContext method() : mode = train/test")

        vector_context = []
        for context in contexts:
            local_context = []
            for utterance in context[-self.config.max_context_length:]: # taking latest (max_context_length) sentences

                if utterance == "":
                    print(context)

                #padding to max_sent_length
                word_indices = self.wordToIndex(utterance)
                word_avg = self.pool_text(word_indices)

                local_context.append(word_avg)

            local_context = np.array(local_context)
            vector_context.append(np.mean(local_context, axis=0))
            

        return np.array(vector_context)



    def oneHotOutput(self, mode=None, size=None):
        '''
        Returns one hot encoding of the output
        '''
        if mode == "train":
            labels = self.toOneHot(self.train_output, size)
        elif mode == "test":
            labels = self.toOneHot(self.test_output, size)
        else:
            print("Set mode properly for toOneHot method() : mode = train/test")
            exit()
        return labels

    def toOneHot(self, data, size=None):
        '''
        Returns one hot label version of data
        '''
        oneHotData = np.zeros((len(data), size))
        oneHotData[range(len(data)),data] = 1
        
        assert(np.array_equal(data, np.argmax(oneHotData, axis=1)))
        return oneHotData

    #### Audio related functions ####

    def getAudioMaxLength(self, data):
        return np.max([feature.shape[1] for feature in data])

    def padAudio(self, data, max_length):

        for ind, instance in enumerate(data):
            if instance.shape[1] < max_length:
                instance = np.concatenate([instance, np.zeros( (instance.shape[0],(max_length-instance.shape[1])))], axis=1)
                data[ind] = instance
            data[ind] = data[ind][:,:max_length]
            data[ind] = data[ind].transpose()
        return np.array(data)


    def getTargetAudio(self, mode=None):

        audio = self.getData(self.TARGET_AUDIO_ID, mode, 
                             "Set mode properly for TargetAudio method() : mode = train/test")

        if mode == "train":
            self.audioMaxLength = self.getAudioMaxLength(audio)

        audio = self.padAudio(audio, self.audioMaxLength)

        if mode == "train":
            self.config.audio_length = audio.shape[1]
            self.config.audio_embedding = audio.shape[2]

        return audio

    def getTargetAudioPool(self, mode=None):

        audio = self.getData(self.TARGET_AUDIO_ID, mode, 
                             "Set mode properly for TargetAudio method() : mode = train/test")

        return np.array([np.mean(feature_vector, axis=1) for feature_vector in audio])


    #### Video related functions ####

    def getTargetVideoPool(self, mode=None):
        video = self.getData(self.TARGET_VIDEO_ID, mode,
                             "Set mode properly for TargetVideo method() : mode = train/test")

        return np.array([np.mean(feature_vector, axis=0) for feature_vector in video])


if __name__ == "__main__":
    dataLoader = DataLoader(config.Config())
