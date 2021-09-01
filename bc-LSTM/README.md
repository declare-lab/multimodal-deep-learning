# Context-Dependent Sentiment Analysis in User-Generated Videos
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

