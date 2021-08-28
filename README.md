# Multimodal Deep Learning

Announcing the multimodal deep learning repository that contains implementation of various deep learning-based models to solve different multimodal problems such as multimodal representation learning, multimodal fusion for downstream tasks e.g., multimodal sentiment analysis.

- [Models](#models)
  * [MISA (PyTorch)](#MISA-Modality--Invariant-and--Specific-Representations-for-Multimodal-Sentiment-Analysis)
  * [BBFN (PyTorch)](#Bi-Bimodal-Modality-Fusion-for-Correlation-Controlled-Multimodal-Sentiment-Analysis)
  * [Multimodal-Infomax (PyTorch)](#Multimodal-Infomax)
  * [Hfusion](#hfusion)
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

# How to run
``python3 hfusion.py``

## Requirements

Keras >= 2.0, Tensorflow >= 1.7, Numpy, Scikit-learn

# Citation

``Majumder, N., Hazarika, D., Gelbukh, A., Cambria, E. and Poria, S., 2018. Multimodal sentiment analysis using hierarchical fusion with context modeling. Knowledge-Based Systems, 161, pp.124-133.``

# Context-Dependent Sentiment Analysis in User Generated Videos
Code for the paper [Context-Dependent Sentiment Analysis in User-Generated Videos](http://sentic.net/context-dependent-sentiment-analysis-in-user-generated-videos.pdf) (ACL 2017).

## NOTE: Here is the updated version of the code - https://github.com/soujanyaporia/multimodal-sentiment-analysis

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
