# Multimodal Deep Learning

Announcing the multimodal deep learning repository that contains implementation of various deep learning-based models to solve different multimodal problems such as multimodal representation learning, multimodal fusion for downstream tasks e.g., multimodal sentiment analysis.

- [Models](#models)
  * [MISA (PyTorch)](#MISA-Modality--Invariant-and--Specific-Representations-for-Multimodal-Sentiment-Analysis)
  * [BBFN (PyTorch)](#Bi-Bimodal-Modality-Fusion-for-Correlation-Controlled-Multimodal-Sentiment-Analysis)
  * [Multimodal-Infomax (PyTorch)](#Multimodal-Infomax)
  * [Hfusion](#hfusion)
  * [contextual-attention-based-LSTM](Attention-based-multimodal-fusion-for-sentiment-analysis)
  * [bc-LSTM](#Context--Dependent-Sentiment-Analysis-in-User-Generated-Videos)
  * [Contextual-Multimodal-Fusion](#Contextual-Inter--modal-Attention-for-Multimodal-Sentiment-Analysis)


# Models

## MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis
Code for the [ACM MM 2020](https://2020.acmmm.org) paper [MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis](https://arxiv.org/pdf/2005.03545.pdf)


<p align="center">
  <img width="600" src="MISA/misa-pic.png">
</p>



### Setup the environment

We work with a conda environment.

```
conda env create -f environment.yml
conda activate misa-code
```

### Data Download

- Install [CMU Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK). Ensure, you can perform ```from mmsdk import mmdatasdk```.    
- Option 1: Download [pre-computed splits](https://drive.google.com/drive/folders/1IBwWNH0XjPnZWaAlP1U2tIJH6Rb3noMI?usp=sharing) and place the contents inside ```datasets``` folder.     
- Option 2: Re-create splits by downloading data from MMSDK. For this, simply run the code as detailed next.

### Running the code

1. ```cd src```
2. Set ```word_emb_path``` in ```config.py``` to [glove file](http://nlp.stanford.edu/data/glove.840B.300d.zip).
3. Set ```sdk_dir``` to the path of CMU-MultimodalSDK.
2. ```python train.py --data mosi```. Replace ```mosi``` with ```mosei``` or ```ur_funny``` for other datasets.

### Citation

If this paper is useful for your research, please cite us at:

```
@article{hazarika2020misa,
  title={MISA: Modality-Invariant and-Specific Representations for Multimodal Sentiment Analysis},
  author={Hazarika, Devamanyu and Zimmermann, Roger and Poria, Soujanya},
  journal={arXiv preprint arXiv:2005.03545},
  year={2020}
}
```

### Contact

For any questions, please email at [hazarika@comp.nus.edu.sg](mailto:hazarika@comp.nus.edu.sg)

## Bi-Bimodal Modality Fusion for Correlation-Controlled Multimodal Sentiment Analysis

This repository contains official implementation of the paper: [Bi-Bimodal Modality Fusion for Correlation-Controlled Multimodal Sentiment Analysis (ICMI 2021)](https://arxiv.org/abs/2107.13669)

### Model Architecture

Overview of our Bi-Bimodal Fusion Network (BBFN). It learns two text-related pairs of representations, text-acoustic and text-visual by enforcing each pair of modalities to complement mutually. Finally, the four (two pairs) head representations are concatenated
to generate the final prediction.

![Alt text](BBFN/img/model2.png?raw=true "Model")

A single complementation layer: two identical pipelines (left and right) propagate the main modality and fuse that
with complementary modality with regularization and gated control.

![Alt text](BBFN/img/singlelayer.png?raw=true "Model")

### Results

Results on the test set of CMU-MOSI and CMU-MOSEI dataset. Notation: △ indicates results in the corresponding line are excerpted from previous papers; † means the results are reproduced with publicly visible source code and applicable hyperparameter setting; ‡ shows the results have experienced paired t-test with 𝑝 < 0.05 and demonstrate significant improvement over MISA, the state-of-the-art model.

![Alt text](BBFN/img/results2.png?raw=true "Model")

### Usage
1. Set up conda environemnt
```
conda env create -f environment.yml
conda activate BBFN
```

2. Install [CMU Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK)

3. Set `sdk_dir` in `src/config.py` to the path of CMU-MultimodalSDK

4. Train the model
```
cd src
python main.py --dataset <dataset_name> --data_path <path_to_dataset>
```
We provide a script `scripts/run.sh` for your reference.

### Citation
Please cite our paper if you find our work useful  for your research:
```bibtex
@article{han2021bi,
  title={Bi-Bimodal Modality Fusion for Correlation-Controlled Multimodal Sentiment Analysis},
  author={Han, Wei and Chen, Hui and Gelbukh, Alexander and Zadeh, Amir and Morency, Louis-philippe and Poria, Soujanya},
  journal={ICMI 2021},
  year={2021}
}
```

### Contact
Should you have any question, feel free to contact me through [henryhan88888@gmail.com](henryhan88888@gmail.com)

# Hfusion
Codes for the paper ``Multimodal sentiment analysis using hierarchical fusion with context modeling``

## How to run
``python3 hfusion.py``

## Requirements

Keras >= 2.0, Tensorflow >= 1.7, Numpy, Scikit-learn

## Citation

``Majumder, N., Hazarika, D., Gelbukh, A., Cambria, E. and Poria, S., 2018. Multimodal sentiment analysis using hierarchical fusion with context modeling. Knowledge-Based Systems, 161, pp.124-133.``

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

# Context-Dependent Sentiment Analysis in User Generated Videos
Code for the paper [Context-Dependent Sentiment Analysis in User-Generated Videos](http://sentic.net/context-dependent-sentiment-analysis-in-user-generated-videos.pdf) (ACL 2017).

### Requirements
Code is written in Python (2.7) and requires Keras (2.0.6) with Theano backend.

### Description
In this paper, we propose a LSTM-based model that enables utterances to capture contextual information from their surroundings in the same video, thus aiding the classification process in multimodal sentiment analysis.

![Alt text](network.jpg?raw=true "Title")

This repository contains the code for the mentioned paper. Each contextual LSTM (Figure 2 in the paper) is implemented as shown in above figure. For more details, please refer to the paper.   
Note: Unlike the paper, we haven't used an SVM on the penultimate layer. This is in effort to keep the whole network differentiable at some performance cost.

### Dataset
We provide results on the [MOSI dataset](https://arxiv.org/pdf/1606.06259.pdf)  
Please cite the creators


### Preprocessing
As data is typically present in utterance format, we combine all the utterances belonging to a video using the following code

```
python create_data.py
```

Note: This will create speaker independent train and test splits

### Running sc-lstm

Sample command:

```
python lstm.py --unimodal True
python lstm.py --unimodal False
```

Note: Keeping the unimodal flag as True (default False) shall train all unimodal lstms first (level 1 of the network mentioned in the paper)

### Citation

If using this code, please cite our work using :
```
@inproceedings{soujanyaacl17,
  title={Context-dependent sentiment analysis in user-generated videos},
  author={Poria, Soujanya  and Cambria, Erik and Hazarika, Devamanyu and Mazumder, Navonil and Zadeh, Amir and Morency, Louis-Philippe},
  booktitle={Association for Computational Linguistics},
  year={2017}
}
```

### Credits

Devamanyu Hazarika, Soujanya Poria

# Contextual Inter-modal Attention for Multimodal Sentiment Analysis
Code for the paper [Contextual Inter-modal Attention for Multi-modal Sentiment Analysis](http://www.aclweb.org/anthology/D18-1382) (EMNLP 2018).

### Dataset
We provide results on the [MOSI dataset](https://arxiv.org/pdf/1606.06259.pdf).  
Please cite the creators.

## Requirements:
Python 3.5  
Keras (Tensorflow backend)  2.2.4  
Scikit-learn 0.20.0  


### Experiments

```
python create_data.py
python trimodal_attention_models.py
```

### Citation

If you use this code in your research, please cite our work using:
```
@inproceedings{ghosal2018contextual,
  title={Contextual Inter-modal Attention for Multi-modal Sentiment Analysis},
  author={Ghosal, Deepanway and Akhtar, Md Shad and Chauhan, Dushyant and Poria, Soujanya and Ekbal, Asif and Bhattacharyya, Pushpak},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={3454--3466},
  year={2018}
}
```

### Credits

Some of the functionalities in this repo are borrowed from https://github.com/soujanyaporia/contextual-utterance-level-multimodal-sentiment-analysis

### Authors

[Deepanway Ghosal](https://github.com/deepanwayx), [Soujanya Poria](https://github.com/soujanyaporia)
