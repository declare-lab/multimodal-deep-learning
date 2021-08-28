import pickle
import sys

import keras
import numpy as np


def createOneHot(train_label, test_label):
    maxlen = int(max(train_label.max(), test_label.max()))

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            train[i, j, train_label[i, j]] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            test[i, j, test_label[i, j]] = 1

    return train, test


def createOneHotMosei3way(train_label, test_label):
    maxlen = 2
    # print(maxlen)

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            if train_label[i, j] > 0:
                train[i, j, 1] = 1
            else:
                if train_label[i, j] < 0:
                    train[i, j, 0] = 1
                else:
                    if train_label[i, j] == 0:
                        train[i, j, 2] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            if test_label[i, j] > 0:
                test[i, j, 1] = 1
            else:
                if test_label[i, j] < 0:
                    test[i, j, 0] = 1
                else:
                    if test_label[i, j] == 0:
                        test[i, j, 2] = 1
    return train, test


def createOneHotMosei2way(train_label, test_label):
    maxlen = 1
    # print(maxlen)

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            if train_label[i, j] > 0:
                train[i, j, 1] = 1
            else:
                if train_label[i, j] <= 0:
                    train[i, j, 0] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            if test_label[i, j] > 0:
                test[i, j, 1] = 1
            else:
                if test_label[i, j] <= 0:
                    test[i, j, 0] = 1

    return train, test


def batch_iter(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def get_raw_data(data, classes):
    if data == 'iemocap':
        return get_iemocap_raw(classes)
    mode = 'audio'
    with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format(data, mode, classes), 'rb') as handle:
        u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        if data == 'mosi':
            (audio_train, train_label, audio_test, test_label, _, train_length, test_length) = u.load()
        elif data == 'mosei':
            (
                audio_train, train_label, _, _, audio_test, test_label, _, train_length, _, test_length, _, _,
                _) = u.load()
            print(test_label.shape)

    mode = 'text'
    with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format(data, mode, classes), 'rb') as handle:
        u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        if data == 'mosi':
            (text_train, train_label, text_test, test_label, _, train_length, test_length) = u.load()
        elif data == 'mosei':
            (text_train, train_label, _, _, text_test, test_label, _, train_length, _, test_length, _, _, _) = u.load()
            print(test_label.shape)

    mode = 'video'
    with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format(data, mode, classes), 'rb') as handle:
        u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        if data == 'mosi':
            (video_train, train_label, video_test, test_label, _, train_length, test_length) = u.load()
        elif data == 'mosei':
            (
                video_train, train_label, _, _, video_test, test_label, _, train_length, _, test_length, _, _,
                _) = u.load()
            print(test_label.shape)

    print('audio_train', audio_train.shape)
    print('audio_test', audio_test.shape)

    train_data = np.concatenate((audio_train, video_train, text_train), axis=-1)
    test_data = np.concatenate((audio_test, video_test, text_test), axis=-1)

    train_label = train_label.astype('int')
    test_label = test_label.astype('int')
    print(train_data.shape)
    print(test_data.shape)
    train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
    for i in range(len(train_length)):
        train_mask[i, :train_length[i]] = 1.0

    test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
    for i in range(len(test_length)):
        test_mask[i, :test_length[i]] = 1.0

    train_label, test_label = createOneHot(train_label, test_label)

    print('train_mask', train_mask.shape)

    seqlen_train = train_length
    seqlen_test = test_length

    return train_data, test_data, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask


def get_iemocap_raw(classes):
    if sys.version_info[0] == 2:
        f = open("dataset/iemocap/raw/IEMOCAP_features_raw.pkl", "rb")
        videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(
            f)

        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
    else:
        f = open("dataset/iemocap/raw/IEMOCAP_features_raw.pkl", "rb")
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = u.load()
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

    # print(len(trainVid))
    # print(len(testVid))

    train_audio = []
    train_text = []
    train_visual = []
    train_seq_len = []
    train_label = []

    test_audio = []
    test_text = []
    test_visual = []
    test_seq_len = []
    test_label = []
    for vid in trainVid:
        train_seq_len.append(len(videoIDs[vid]))
    for vid in testVid:
        test_seq_len.append(len(videoIDs[vid]))

    max_len = max(max(train_seq_len), max(test_seq_len))
    print('max_len', max_len)
    for vid in trainVid:
        train_label.append(videoLabels[vid] + [0] * (max_len - len(videoIDs[vid])))
        pad = [np.zeros(videoText[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        text = np.stack(videoText[vid] + pad, axis=0)
        train_text.append(text)

        pad = [np.zeros(videoAudio[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        audio = np.stack(videoAudio[vid] + pad, axis=0)
        train_audio.append(audio)

        pad = [np.zeros(videoVisual[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        video = np.stack(videoVisual[vid] + pad, axis=0)
        train_visual.append(video)

    for vid in testVid:
        test_label.append(videoLabels[vid] + [0] * (max_len - len(videoIDs[vid])))
        pad = [np.zeros(videoText[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        text = np.stack(videoText[vid] + pad, axis=0)
        test_text.append(text)

        pad = [np.zeros(videoAudio[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        audio = np.stack(videoAudio[vid] + pad, axis=0)
        test_audio.append(audio)

        pad = [np.zeros(videoVisual[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        video = np.stack(videoVisual[vid] + pad, axis=0)
        test_visual.append(video)

    train_text = np.stack(train_text, axis=0)
    train_audio = np.stack(train_audio, axis=0)
    train_visual = np.stack(train_visual, axis=0)
    # print(train_text.shape)
    # print(train_audio.shape)
    # print(train_visual.shape)

    # print()
    test_text = np.stack(test_text, axis=0)
    test_audio = np.stack(test_audio, axis=0)
    test_visual = np.stack(test_visual, axis=0)
    # print(test_text.shape)
    # print(test_audio.shape)
    # print(test_visual.shape)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    train_seq_len = np.array(train_seq_len)
    test_seq_len = np.array(test_seq_len)
    # print(train_label.shape)
    # print(test_label.shape)
    # print(train_seq_len.shape)
    # print(test_seq_len.shape)

    train_mask = np.zeros((train_text.shape[0], train_text.shape[1]), dtype='float')
    for i in range(len(train_seq_len)):
        train_mask[i, :train_seq_len[i]] = 1.0

    test_mask = np.zeros((test_text.shape[0], test_text.shape[1]), dtype='float')
    for i in range(len(test_seq_len)):
        test_mask[i, :test_seq_len[i]] = 1.0

    train_label, test_label = createOneHot(train_label, test_label)

    train_data = np.concatenate((train_audio, train_visual, train_text), axis=-1)
    test_data = np.concatenate((test_audio, test_visual, test_text), axis=-1)

    return train_data, test_data, train_audio, test_audio, train_text, test_text, train_visual, test_visual, train_label, test_label, train_seq_len, test_seq_len, train_mask, test_mask


def get_raw_data_iemocap(data, classes):
    videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(
        open("./dataset/iemocap/raw/IEMOCAP_features_raw.pkl", 'rb'), encoding='latin1')

    train_data = []
    test_data = []
    train_label = []
    test_label = []
    train_length = []
    test_length = []
    audio_train = []
    video_train = []
    text_train = []
    audio_test = []
    video_test = []
    text_test = []

    for vid in trainVid:
        text_train.append(videoText[vid])
        audio_train.append(videoAudio[vid])
        video_train.append(videoVisual[vid])
        train_label.append(videoLabels[vid])
        train_length.append(len(videoLabels[vid]))
    for vid in testVid:
        text_test.append(videoText[vid])
        audio_test.append(videoAudio[vid])
        video_test.append(videoVisual[vid])
        test_label.append(videoLabels[vid])
        test_length.append(len(videoLabels[vid]))

    text_train = keras.preprocessing.sequence.pad_sequences(text_train, maxlen=110, padding='post', dtype='float32')
    audio_train = keras.preprocessing.sequence.pad_sequences(audio_train, maxlen=110, padding='post', dtype='float32')
    video_train = keras.preprocessing.sequence.pad_sequences(video_train, maxlen=110, padding='post', dtype='float32')
    text_test = keras.preprocessing.sequence.pad_sequences(text_test, maxlen=110, padding='post', dtype='float32')
    audio_test = keras.preprocessing.sequence.pad_sequences(audio_test, maxlen=110, padding='post', dtype='float32')
    video_test = keras.preprocessing.sequence.pad_sequences(video_test, maxlen=110, padding='post', dtype='float32')

    train_label = keras.preprocessing.sequence.pad_sequences(train_label, maxlen=110, padding='post', dtype='int32')
    test_label = keras.preprocessing.sequence.pad_sequences(test_label, maxlen=110, padding='post', dtype='int32')
    # print(text_train[0, -1, :])
    # print(audio_train[0, -1, :])
    # print(video_train[0, -1, :])
    train_mask = np.zeros((text_train.shape[0], text_train.shape[1]), dtype='float')
    for i in range(len(train_length)):
        train_mask[i, :train_length[i]] = 1.0

    test_mask = np.zeros((text_test.shape[0], text_test.shape[1]), dtype='float')
    for i in range(len(test_length)):
        test_mask[i, :test_length[i]] = 1.0

    train_label, test_label = createOneHot(train_label, test_label)

    train_data = np.concatenate((audio_train, video_train, text_train), axis=-1)
    test_data = np.concatenate((audio_test, video_test, text_test), axis=-1)

    seqlen_train = train_length
    seqlen_test = test_length

    return train_data, test_data, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask
