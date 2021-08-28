# Bi-Bimodal Modality Fusion for Correlation-Controlled Multimodal Sentiment Analysis

This repository contains official implementation of the paper: [Bi-Bimodal Modality Fusion for Correlation-Controlled Multimodal Sentiment Analysis (ICMI 2021)](https://arxiv.org/abs/2107.13669)

## Model Architecture

Overview of our Bi-Bimodal Fusion Network (BBFN). It learns two text-related pairs of representations, text-acoustic and text-visual by enforcing each pair of modalities to complement mutually. Finally, the four (two pairs) head representations are concatenated
to generate the final prediction.

![Alt text](img/model2.png?raw=true "Model")

A single complementation layer: two identical pipelines (left and right) propagate the main modality and fuse that
with complementary modality with regularization and gated control.

![Alt text](img/singlelayer.png?raw=true "Model")

## Results

Results on the test set of CMU-MOSI and CMU-MOSEI dataset. Notation: △ indicates results in the corresponding line are excerpted from previous papers; † means the results are reproduced with publicly visible source code and applicable hyperparameter setting; ‡ shows the results have experienced paired t-test with 𝑝 < 0.05 and demonstrate significant improvement over MISA, the state-of-the-art model.

![Alt text](img/results2.png?raw=true "Model")

## Usage
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

## Citation
Please cite our paper if you find our work useful  for your research:
```bibtex
@article{han2021bi,
  title={Bi-Bimodal Modality Fusion for Correlation-Controlled Multimodal Sentiment Analysis},
  author={Han, Wei and Chen, Hui and Gelbukh, Alexander and Zadeh, Amir and Morency, Louis-philippe and Poria, Soujanya},
  journal={ICMI 2021},
  year={2021}
}
```

## Contact 
Should you have any question, feel free to contact me through [henryhan88888@gmail.com](henryhan88888@gmail.com)

