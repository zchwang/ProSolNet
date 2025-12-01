#!/bin/bash  

script_dir="$(cd "$(dirname "$0")" && pwd)"
dpath=${script_dir}/../../generate_features/ProSolNet/outputs

python train.py \
    -global_fpath ${dpath}/protein_global_feats.pkl \
    -struc_fpath ${dpath}/protein_3d_info.npy \
    -surf_fpath ${dpath}/protein_surface_feats.npy \
    -lm_fpath ${dpath}/protein_prott5_embedding.npy \
    -split_fpath data-split_dict.npy \
    -label_fpath all_labels.csv \
    -lr 0.001 \
    -batch_size 64 \
    -hid_dim 64 \
    -epoch 15 \
    -device cuda:0
