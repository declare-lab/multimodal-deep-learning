## Contextual Inter-modal Attention for Multi-modal Sentiment Analysis
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