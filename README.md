# Low-rank-Multimodal-Fusion

This is the repository for "Efficient Low-rank Multimodal Fusion with Modality-Specific Factors", Liu and Shen, et. al. ACL 2018.

## Dependencies

Python 2.7 (now experimentally has Python 3.6+ support)

```
torch=0.3.1
sklearn
numpy
```

You can install the libraries via `python -m pip install -r requirements.txt`.


## Data for Experiments

The processed data for the experiments (CMU-MOSI, IEMOCAP, POM) can be downloaded here:

https://drive.google.com/open?id=1CixSaw3dpHESNG0CaCJV6KutdlANP_cr

To run the code, you should download the pickled datasets and put them in the `data` directory.

Note that there might be NaN values in acoustic features, you could replace them with 0s.

## Training Your Model

To run the code for experiments (grid search), use the scripts `train_xxx.py`. They have some commandline arguments as listed here:

```
`--run_id`: an user-specified unique ID to ensure that saved results/models don't override each other.

`--epochs`: the number of maximum epochs in training. Since early-stopping is used to prevent overfitting, in actual training the number of epochs could be less than what you specify here.

`--patience`: if the model performance does not improve in `--patience` many validation evaluations consecutively, the training will early-stop.

`output_dim`: output dimension of the model. Default value in each script should work.

`signiture`: an optional string that's added to the output file name. Intended to use as some sort of comment.

`cuda`: whether or not to use GPU in training. If not specified, will use CPU.

`data_path`: the path to the data directory. Defaults to './data', but if you prefer storing the data else where you can change this.

`model_path`: the path to the directory where models will be saved.

`output_path`: the path to the directory where the grid search results will be saved.

`max_len`: the maximum length of training data sequences. Longer/shorter sequences will be truncated/padded.

`emotion`: (exclusive for IEMOCAP) specifies which emotion category you want to train the model to predict. Can be 'happy', 'sad', 'angry', 'neutral'.
```

An example would be

`python train_mosi.py --run_id 19260817 --epochs 50 --patience 20 --output_dim 1 --signiture test_run_big_model`

## Hyperparameters

Some hyper parameters for reproducing the results in the paper are in the `hyperparams.txt` file.
