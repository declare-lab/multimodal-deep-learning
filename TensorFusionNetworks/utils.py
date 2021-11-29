'''
This script does all the data preprocessing.
You'll need to install CMU-Multimodal DataSDK 
(https://github.com/A2Zadeh/CMU-MultimodalDataSDK) to use this script.
There's a packaged (and more up-to-date) version
of the utils below at https://github.com/Justin1904/tetheras-utils.
Preprocessing multimodal data is really tiring...
'''
from __future__ import print_function
import mmdata
import numpy as np
from torch.utils.data import Dataset

def pad(data, max_len):
    """Pads data without time stamps"""
    data = remove_timestamps(data)
    n_rows = data.shape[0]
    dim = data.shape[1]
    if max_len >= n_rows:
        diff = max_len - n_rows
        padding = np.zeros((diff, dim))
        padded = np.concatenate((padding, data))
        return padded
    else:
        return data[-max_len:]

def remove_timestamps(segment_data):
    """Removes the start and end time stamps in the Multimodal Data SDK"""
    return np.array([feature[2] for feature in segment_data])

class ProcessedDataset(Dataset):
    """The class object for processed data, pipelined from CMU-MultimodalDataSDK through MultimodalDataset"""
    def __init__(self, audio, visual, text, labels):
        self.audio = audio
        self.visual = visual
        self.text = text
        self.labels = labels

    def __len__(self):
        """Checks the number of data points are the same across different modalities, and return length"""
        assert self.audio.shape[1] == self.visual.shape[1] and self.visual.shape[1] == self.text.shape[1] and self.text.shape[1] == self.labels.shape[0]
        return self.audio.shape[1]

    def __getitem__(self, idx):
        """Returns the target element by index"""
        return [self.audio[:, idx, :], self.visual[:, idx, :], self.text[:, idx, :], self.labels[idx]]


class MultimodalDataset(object):
    """The class object for all multimodal datasets from CMU-MultimodalDataSDK"""
    def __init__(self, dataset, visual='facet', audio='covarep', text='embeddings', pivot='words', sentiments=True, emotions=False, max_len=20):
        # instantiate a multimodal dataloader
        self.dataloader = mmdata.__dict__[dataset]()
        self.max_len = max_len

        # load the separate modalities, it's silly to access parent class' methods
        self.visual = self.dataloader.__class__.__bases__[0].__dict__[visual](self.dataloader)
        self.audio = self.dataloader.__class__.__bases__[0].__dict__[audio](self.dataloader)
        self.text = self.dataloader.__class__.__bases__[0].__dict__[text](self.dataloader)
        # self.pivot = self.dataloader.__class__.__bases__[0].__dict__[pivot](self.dataloader)

        # load train/dev/test splits and labels
        self.train_vids = self.dataloader.train()
        self.valid_vids = self.dataloader.valid()
        self.test_vids = self.dataloader.test()
        if sentiments:
            self.sentiments = self.dataloader.sentiments()
        if emotions:
            self.emotions = self.dataloader.emotions()

        # merge them one by one
        self.dataset = mmdata.Dataset.merge(self.visual, self.text)
        self.dataset = mmdata.Dataset.merge(self.audio, self.dataset)

        # align the modalities
        self.aligned = self.dataset.align(text)
        
        # split the training, validation and test sets and preprocess them
        train_set_ids = []
        for vid in self.train_vids:
            for sid in self.dataset[text][vid].keys():
                if self.triple_check(vid, sid, audio, visual, text):
                    train_set_ids.append((vid, sid))

        valid_set_ids = []
        for vid in self.valid_vids:
            for sid in self.dataset[text][vid].keys():
                if self.triple_check(vid, sid, audio, visual, text):
                    valid_set_ids.append((vid, sid))

        test_set_ids = []
        for vid in self.test_vids:
            for sid in self.dataset[text][vid].keys():
                if self.triple_check(vid, sid, audio, visual, text):
                    test_set_ids.append((vid, sid))

        self.train_set_audio = np.stack([pad(self.aligned[audio][vid][sid], self.max_len) for (vid, sid) in train_set_ids if self.aligned[audio][vid][sid]], axis=1)
        self.valid_set_audio = np.stack([pad(self.aligned[audio][vid][sid], self.max_len) for (vid, sid) in valid_set_ids if self.aligned[audio][vid][sid]], axis=1)
        self.test_set_audio = np.stack([pad(self.aligned[audio][vid][sid], self.max_len) for (vid, sid) in test_set_ids if self.aligned[audio][vid][sid]], axis=1)

        self.train_set_audio = self.validify(self.train_set_audio)
        self.valid_set_audio = self.validify(self.valid_set_audio)
        self.test_set_audio = self.validify(self.test_set_audio)

        self.train_set_visual = np.stack([pad(self.aligned[visual][vid][sid], self.max_len) for (vid, sid) in train_set_ids], axis=1)
        self.valid_set_visual = np.stack([pad(self.aligned[visual][vid][sid], self.max_len) for (vid, sid) in valid_set_ids], axis=1)
        self.test_set_visual = np.stack([pad(self.aligned[visual][vid][sid], self.max_len) for (vid, sid) in test_set_ids], axis=1)

        self.train_set_visual = self.validify(self.train_set_visual)
        self.valid_set_visual = self.validify(self.valid_set_visual)
        self.test_set_visual = self.validify(self.test_set_visual)

        self.train_set_text = np.stack([pad(self.aligned[text][vid][sid], self.max_len) for (vid, sid) in train_set_ids], axis=1)
        self.valid_set_text = np.stack([pad(self.aligned[text][vid][sid], self.max_len) for (vid, sid) in valid_set_ids], axis=1)
        self.test_set_text = np.stack([pad(self.aligned[text][vid][sid], self.max_len) for (vid, sid) in test_set_ids], axis=1)

        self.train_set_text = self.validify(self.train_set_text)
        self.valid_set_text = self.validify(self.valid_set_text)
        self.test_set_text = self.validify(self.test_set_text)

        self.train_set_labels = np.array([self.sentiments[vid][sid] for (vid, sid) in train_set_ids])
        self.valid_set_labels = np.array([self.sentiments[vid][sid] for (vid, sid) in valid_set_ids])
        self.test_set_labels = np.array([self.sentiments[vid][sid] for (vid, sid) in test_set_ids])

        self.train_set_labels = self.validify(self.train_set_labels)
        self.valid_set_labels = self.validify(self.valid_set_labels)
        self.test_set_labels = self.validify(self.test_set_labels)

        self.train_set = ProcessedDataset(self.train_set_audio, self.train_set_visual, self.train_set_text, self.train_set_labels)
        self.valid_set = ProcessedDataset(self.valid_set_audio, self.valid_set_visual, self.valid_set_text, self.valid_set_labels)
        self.test_set = ProcessedDataset(self.test_set_audio, self.test_set_visual, self.test_set_text, self.test_set_labels)

    def triple_check(self, vid, sid, audio, visual, text):
        """Checks if this segment data is intact"""
        if self.aligned[audio][vid][sid] and self.aligned[visual][vid][sid] and self.aligned[text][vid][sid]:
            return True
        else:
            print("Video {} segment {} has incomplete data and has been discarded!".format(vid, sid))
            return False

    def validify(self, array, dummy=0):
        """Check and remove NaN values in the data!"""
        array[array != array] = dummy
        return array

