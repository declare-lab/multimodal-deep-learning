#!/bin/bash

GPU_NO=$1

if [ -z $GPU_NO ];then
    GPU_NO=0
fi

CUDA_VISIBLE_DEVICES=$GPU_NO python main.py --dataset mosi --data_path ./datasets/MOSI