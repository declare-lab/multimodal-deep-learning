import sys
import mmsdk
import os
import re
import pickle
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from mmsdk import mmdatasdk as md
from subprocess import check_call, CalledProcessError

import torch
import torch.nn as nn


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']


# turn off the word2id - define a named function here to allow for pickling
def return_unk():
    return UNK


def load_emb(w2i, path_to_embedding, embedding_size=300, embedding_vocab=2196017, init_emb=None):
    if init_emb is None:
        emb_mat = np.random.randn(len(w2i), embedding_size)
    else:
        emb_mat = init_emb
    f = open(path_to_embedding, 'r')
    found = 0
    for line in tqdm_notebook(f, total=embedding_vocab):
        content = line.strip().split()
        vector = np.asarray(list(map(lambda x: float(x), content[-300:])))
        word = ' '.join(content[:-300])
        if word in w2i:
            idx = w2i[word]
            emb_mat[idx, :] = vector
            found += 1
    print(f"Found {found} words in the embedding file.")
    return torch.tensor(emb_mat).float()





class MOSI:
    def __init__(self, config):

        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))
        
        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = torch.load(CACHE_PATH)

        except:

            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)


            # download highlevel features, low-level (raw) data and labels for the dataset MOSI
            # if the files are already present, instead of downloading it you just load it yourself.
            # here we use CMU_MOSI dataset as example.
            DATASET = md.cmu_mosi
            try:
                md.mmdataset(DATASET.highlevel, DATA_PATH)
            except RuntimeError:
                print("High-level features have been downloaded previously.")

            try:
                md.mmdataset(DATASET.raw, DATA_PATH)
            except RuntimeError:
                print("Raw data have been downloaded previously.")
                
            try:
                md.mmdataset(DATASET.labels, DATA_PATH)
            except RuntimeError:
                print("Labels have been downloaded previously.")
            
            # define your different modalities - refer to the filenames of the CSD files
            visual_field = 'CMU_MOSI_VisualFacet_4.1'
            acoustic_field = 'CMU_MOSI_COVAREP'
            text_field = 'CMU_MOSI_TimestampedWords'


            features = [
                text_field, 
                visual_field, 
                acoustic_field
            ]

            recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
            print(recipe)
            dataset = md.mmdataset(recipe)

            # we define a simple averaging function that does not depend on intervals
            def avg(intervals: np.array, features: np.array) -> np.array:
                try:
                    return np.average(features, axis=0)
                except:
                    return features

            # first we align to words with averaging, collapse_function receives a list of functions
            dataset.align(text_field, collapse_functions=[avg])

            label_field = 'CMU_MOSI_Opinion_Labels'

            # we add and align to lables to obtain labeled segments
            # this time we don't apply collapse functions so that the temporal sequences are preserved
            label_recipe = {label_field: os.path.join(DATA_PATH, label_field + '.csd')}
            dataset.add_computational_sequences(label_recipe, destination=None)
            dataset.align(label_field)

            # obtain the train/dev/test splits - these splits are based on video IDs
            train_split = DATASET.standard_folds.standard_train_fold
            dev_split = DATASET.standard_folds.standard_valid_fold
            test_split = DATASET.standard_folds.standard_test_fold


            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            # define a regular expression to extract the video ID out of the keys
            pattern = re.compile('(.*)\[.*\]')
            num_drop = 0 # a counter to count how many data points went into some processing issues

            for segment in dataset[label_field].keys():
                
                # get the video ID and the features out of the aligned dataset
                vid = re.search(pattern, segment).group(1)
                label = dataset[label_field][segment]['features']
                _words = dataset[text_field][segment]['features']
                _visual = dataset[visual_field][segment]['features']
                _acoustic = dataset[acoustic_field][segment]['features']

                # if the sequences are not same length after alignment, there must be some problem with some modalities
                # we should drop it or inspect the data again
                if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
                    print(f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_acoustic.shape}")
                    num_drop += 1
                    continue

                # remove nan values
                label = np.nan_to_num(label)
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                # remove speech pause tokens - this is in general helpful
                # we should remove speech pauses and corresponding visual/acoustic features together
                # otherwise modalities would no longer be aligned
                actual_words = []
                words = []
                visual = []
                acoustic = []
                for i, word in enumerate(_words):
                    if word[0] != b'sp':
                        actual_words.append(word[0].decode('utf-8'))
                        words.append(word2id[word[0].decode('utf-8')]) # SDK stores strings as bytes, decode into strings here
                        visual.append(_visual[i, :])
                        acoustic.append(_acoustic[i, :])

                words = np.asarray(words)
                visual = np.asarray(visual)
                acoustic = np.asarray(acoustic)


                # z-normalization per instance and remove nan/infs
                visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
                acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

                if vid in train_split:
                    train.append(((words, visual, acoustic, actual_words), label, segment))
                elif vid in dev_split:
                    dev.append(((words, visual, acoustic, actual_words), label, segment))
                elif vid in test_split:
                    test.append(((words, visual, acoustic, actual_words), label, segment))
                else:
                    print(f"Found video that doesn't belong to any splits: {vid}")

            print(f"Total number of {num_drop} datapoints have been dropped.")

            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
            torch.save((pretrained_emb, word2id), CACHE_PATH)

            # Save pickles
            to_pickle(train, DATA_PATH + '/train.pkl')
            to_pickle(dev, DATA_PATH + '/dev.pkl')
            to_pickle(test, DATA_PATH + '/test.pkl')

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "dev":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()




class MOSEI:
    def __init__(self, config):

        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))
        
        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = torch.load(CACHE_PATH)

        except:

            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)


            # download highlevel features, low-level (raw) data and labels for the dataset MOSEI
            # if the files are already present, instead of downloading it you just load it yourself.
            DATASET = md.cmu_mosei
            try:
                md.mmdataset(DATASET.highlevel, DATA_PATH)
            except RuntimeError:
                print("High-level features have been downloaded previously.")

            try:
                md.mmdataset(DATASET.raw, DATA_PATH)
            except RuntimeError:
                print("Raw data have been downloaded previously.")
                
            try:
                md.mmdataset(DATASET.labels, DATA_PATH)
            except RuntimeError:
                print("Labels have been downloaded previously.")
            
            # define your different modalities - refer to the filenames of the CSD files
            visual_field = 'CMU_MOSEI_VisualFacet42'
            acoustic_field = 'CMU_MOSEI_COVAREP'
            text_field = 'CMU_MOSEI_TimestampedWords'


            features = [
                text_field, 
                visual_field, 
                acoustic_field
            ]

            recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
            print(recipe)
            dataset = md.mmdataset(recipe)

            # we define a simple averaging function that does not depend on intervals
            def avg(intervals: np.array, features: np.array) -> np.array:
                try:
                    return np.average(features, axis=0)
                except:
                    return features

            # first we align to words with averaging, collapse_function receives a list of functions
            dataset.align(text_field, collapse_functions=[avg])

            label_field = 'CMU_MOSEI_LabelsSentiment'

            # we add and align to lables to obtain labeled segments
            # this time we don't apply collapse functions so that the temporal sequences are preserved
            label_recipe = {label_field: os.path.join(DATA_PATH, label_field + '.csd')}
            dataset.add_computational_sequences(label_recipe, destination=None)
            dataset.align(label_field)

            # obtain the train/dev/test splits - these splits are based on video IDs
            train_split = DATASET.standard_folds.standard_train_fold
            dev_split = DATASET.standard_folds.standard_valid_fold
            test_split = DATASET.standard_folds.standard_test_fold


            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            # define a regular expression to extract the video ID out of the keys
            pattern = re.compile('(.*)\[.*\]')
            num_drop = 0 # a counter to count how many data points went into some processing issues

            for segment in dataset[label_field].keys():
                
                # get the video ID and the features out of the aligned dataset
                try:
                    vid = re.search(pattern, segment).group(1)
                    label = dataset[label_field][segment]['features']
                    _words = dataset[text_field][segment]['features']
                    _visual = dataset[visual_field][segment]['features']
                    _acoustic = dataset[acoustic_field][segment]['features']
                except:
                    continue

                # if the sequences are not same length after alignment, there must be some problem with some modalities
                # we should drop it or inspect the data again
                if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
                    print(f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_acoustic.shape}")
                    num_drop += 1
                    continue

                # remove nan values
                label = np.nan_to_num(label)
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                # remove speech pause tokens - this is in general helpful
                # we should remove speech pauses and corresponding visual/acoustic features together
                # otherwise modalities would no longer be aligned
                actual_words = []
                words = []
                visual = []
                acoustic = []
                for i, word in enumerate(_words):
                    if word[0] != b'sp':
                        actual_words.append(word[0].decode('utf-8'))
                        words.append(word2id[word[0].decode('utf-8')]) # SDK stores strings as bytes, decode into strings here
                        visual.append(_visual[i, :])
                        acoustic.append(_acoustic[i, :])

                words = np.asarray(words)
                visual = np.asarray(visual)
                acoustic = np.asarray(acoustic)

                # z-normalization per instance and remove nan/infs
                visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
                acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

                if vid in train_split:
                    train.append(((words, visual, acoustic, actual_words), label, segment))
                elif vid in dev_split:
                    dev.append(((words, visual, acoustic, actual_words), label, segment))
                elif vid in test_split:
                    test.append(((words, visual, acoustic, actual_words), label, segment))
                else:
                    print(f"Found video that doesn't belong to any splits: {vid}")
                

            print(f"Total number of {num_drop} datapoints have been dropped.")

            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
            torch.save((pretrained_emb, word2id), CACHE_PATH)

            # Save pickles
            to_pickle(train, DATA_PATH + '/train.pkl')
            to_pickle(dev, DATA_PATH + '/dev.pkl')
            to_pickle(test, DATA_PATH + '/test.pkl')

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "dev":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()




class UR_FUNNY:
    def __init__(self, config):

        
        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = torch.load(CACHE_PATH)

        except:


            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)


            data_folds=load_pickle(DATA_PATH + '/data_folds.pkl')
            train_split=data_folds['train']
            dev_split=data_folds['dev']
            test_split=data_folds['test']

            

            word_aligned_openface_sdk=load_pickle(DATA_PATH + "/openface_features_sdk.pkl")
            word_aligned_covarep_sdk=load_pickle(DATA_PATH + "/covarep_features_sdk.pkl")
            word_embedding_idx_sdk=load_pickle(DATA_PATH + "/word_embedding_indexes_sdk.pkl")
            word_list_sdk=load_pickle(DATA_PATH + "/word_list.pkl")
            humor_label_sdk = load_pickle(DATA_PATH + "/humor_label_sdk.pkl")

            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            num_drop = 0 # a counter to count how many data points went into some processing issues

            # Iterate over all possible utterances
            for key in humor_label_sdk.keys():

                label = np.array(humor_label_sdk[key], dtype=int)
                _word_id = np.array(word_embedding_idx_sdk[key]['punchline_embedding_indexes'])
                _acoustic = np.array(word_aligned_covarep_sdk[key]['punchline_features'])
                _visual = np.array(word_aligned_openface_sdk[key]['punchline_features'])


                if not _word_id.shape[0] == _acoustic.shape[0] == _visual.shape[0]:
                    num_drop += 1
                    continue

                # remove nan values
                label = np.array([np.nan_to_num(label)])[:, np.newaxis]
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)


                actual_words = []
                words = []
                visual = []
                acoustic = []
                for i, word_id in enumerate(_word_id):
                    word = word_list_sdk[word_id]
                    actual_words.append(word)
                    words.append(word2id[word])
                    visual.append(_visual[i, :])
                    acoustic.append(_acoustic[i, :])

                words = np.asarray(words)
                visual = np.asarray(visual)
                acoustic = np.asarray(acoustic)

                # z-normalization per instance and remove nan/infs
                visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
                acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

                if key in train_split:
                    train.append(((words, visual, acoustic, actual_words), label))
                elif key in dev_split:
                    dev.append(((words, visual, acoustic, actual_words), label))
                elif key in test_split:
                    test.append(((words, visual, acoustic, actual_words), label))
                else:
                    print(f"Found video that doesn't belong to any splits: {key}")

            print(f"Total number of {num_drop} datapoints have been dropped.")
            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
            torch.save((pretrained_emb, word2id), CACHE_PATH)

            # Save pickles
            to_pickle(train, DATA_PATH + '/train.pkl')
            to_pickle(dev, DATA_PATH + '/dev.pkl')
            to_pickle(test, DATA_PATH + '/test.pkl')

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "dev":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()