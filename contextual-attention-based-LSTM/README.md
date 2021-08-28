# Attention-based multimodal fusion for sentiment analysis
Attention-based multimodal fusion for sentiment analysis

Code for the paper

[Context-Dependent Sentiment Analysis in User-Generated Videos](http://sentic.net/context-dependent-sentiment-analysis-in-user-generated-videos.pdf) (ACL 2017).

[Multi-level Multiple Attentions for Contextual Multimodal Sentiment Analysis](https://ieeexplore.ieee.org/abstract/document/8215597/)(ICDM 2017).

![Alt text](atlstm3.jpg?raw=true "The attention based fusion mechanism (ICDM 2017)")


### Preprocessing
**Edit:** the create_data.py is obsolete. The pre-processed datasets have already been provided in the dataset/ folder in the repo. Use them directly.

As data is typically present in utterance format, we combine all the utterances belonging to a video using the following code

```
python create_data.py
```

Note: This will create speaker independent train and test splits
In dataset/mosei, extract the zip into a folder named 'raw'.
Also, extract 'unimodal_mosei_3way.pickle.zip'

### Running the model

Sample command:

With attention-based fusion:
```
python run.py --unimodal True --fusion True
python run.py --unimodal False --fusion True
```
Without attention-based and with concatenation-based fusion:
```
python run.py --unimodal True --fusion False
python run.py --unimodal False --fusion False
```
Utterance level attention:
```
python run.py --unimodal False --fusion True --attention_2 True
python run.py --unimodal False --fusion True --attention_2 True
```
Note:
1. Keeping the unimodal flag as True (default False) shall train all unimodal lstms first (level 1 of the network mentioned in the paper)
2. Setting --fusion True applies only to multimodal network.

### Datasets:
We provide results on the [MOSI](https://arxiv.org/pdf/1606.06259.pdf), [MOSEI ](http://aclweb.org/anthology/P18-1208) and [IEMOCAP](https://sail.usc.edu/iemocap/) datasets.<br>
Please cite the creators.

We are adding more datasets, stay tuned.

Use ```--data [mosi|mosei|iemocap] and --classes [2|3|6]``` in the above commands to test different configurations on different datasets.

mosi: 2 classes<br>
mosei: 3 classes<br>
iemocap: 6 classes<br>

Example: 
```
python run.py --unimodal False --fusion True --attention_2 True --data mosei --classes 3
```

#### Dataset details
##### MOSI:
2 classes: Positive/Negative <br>
Raw Features: (Pickle files) <br>
Audio: dataset/mosi/raw/audio_2way.pickle <br>
Text: dataset/mosi/raw/text_2way.pickle <br>
Video: dataset/mosi/raw/video_2way.pickle <br>

**Each file contains: <br>**
train_data, train_label, test_data, test_label, maxlen, train_length, test_length

train_data - np.array of dim (62, 63, feature_dim) <br>
train_label - np.array of dim (62, 63, 2) <br>
test_data - np.array of dim (31, 63, feature_dim) <br>
test_label - np.array of dim (31, 63, 2) <br>
maxlen - max utterance length  int of value 63 <br>
train_length - utterance length of each video in train data. <br>
test_length - utterance length of each video in test data. <br>

Train/Test split: 62/31 videos. Each video has utterances. The videos are padded to 63 utterances.

##### IEMOCAP:
6 classes: happy/sad/neutral/angry/excited/frustrated<br>
Raw Features: dataset/iemocap/raw/IEMOCAP_features_raw.pkl (Pickle files) <br>
The file contains:  
videoIDs[vid] = List of utterance IDs in this video in the order of occurance <br>
videoSpeakers[vid] = List of speaker turns. e.g. [M, M, F, M, F]. here M = Male, F = Female <br>
videoText[vid] = List of textual features for each utterance in video vid. <br>
videoAudio[vid] = List of audio features for each utterance in video vid. <br>
videoVisual[vid] = List of visual features for each utterance in video vid. <br>
videoLabels[vid] = List of label indices for each utterance in video vid. <br>
videoSentence[vid] = List of sentences for each utterance in video vid. <br>
trainVid =  List of videos (videos IDs) in train set. <br>
testVid =  List of videos (videos IDs) in test set. <br>

Refer to the file dataset/iemocap/raw/loadIEMOCAP.py for more information.
We use this data to create a speaker independent train and test splits in the format. (videos x utterances x features)

Train/Test split: 120/31 videos. Each video has utterances. The videos are padded to 110 utterances.

##### MOSEI:
3 classes: positive/negative/neutral <br>
Raw Features: (Pickle files) <br>
Audio: dataset/mosei/raw/audio_3way.pickle <br>
Text: dataset/mosei/raw/text_3way.pickle <br>
Video: dataset/mosei/raw/video_3way.pickle <br>

The file contains:
train_data, train_label, test_data, test_label, maxlen, train_length, test_length

train_data - np.array of dim (2250, 98, feature_dim) <br>
train_label - np.array of dim (62, 63, 2) <br>
test_data - np.array of dim (31, 63, feature_dim) <br>
test_label - np.array of dim (31, 63, 2) <br>
maxlen - max utterance length  int of value 98 <br>
train_length - utterance length of each video in train data. <br>
test_length - utterance length of each video in test data. <br>

Train/Test split: 2250/678 videos. Each video has utterances. The videos are padded to 98 utterances.


### Citation 

If using this code, please cite our work using : 
```
@inproceedings{soujanyaacl17,
  title={Context-dependent sentiment analysis in user-generated videos},
  author={Poria, Soujanya  and Cambria, Erik and Hazarika, Devamanyu and Mazumder, Navonil and Zadeh, Amir and Morency, Louis-Philippe},
  booktitle={Association for Computational Linguistics},
  year={2017}
}

@inproceedings{poriaicdm17, 
author={S. Poria and E. Cambria and D. Hazarika and N. Mazumder and A. Zadeh and L. P. Morency}, 
booktitle={2017 IEEE International Conference on Data Mining (ICDM)}, 
title={Multi-level Multiple Attentions for Contextual Multimodal Sentiment Analysis}, 
year={2017},  
pages={1033-1038}, 
keywords={data mining;feature extraction;image classification;image fusion;learning (artificial intelligence);sentiment analysis;attention-based networks;context learning;contextual information;contextual multimodal sentiment;dynamic feature fusion;multilevel multiple attentions;multimodal sentiment analysis;recurrent model;utterances;videos;Context modeling;Feature extraction;Fuses;Sentiment analysis;Social network services;Videos;Visualization}, 
doi={10.1109/ICDM.2017.134}, 
month={Nov},}
```

### Credits

[Soujanya Poria](http://sporia.info/)

[Gangeshwar Krishnamurthy](http://www.gangeshwark.com/) (gangeshwark@gmail.com; Github: @gangeshwark)
