#!/bin/bash  

script_dir="$(cd "$(dirname "$0")" && pwd)"
dpath=${script_dir}/../../generate_features/ProSolNet_mut/outputs

python train.py \
    -struc_fpath ${dpath}/protein_3d_info.npy \
    -surf_fpath ${dpath}/protein_surface_feats.npy \
    -lm_fpath ${dpath}/protein_prott5_embedding.npy \
    -label_fpath all_labels.npy \
    -split_fpath data-split_dict.npy \
    -batch_size 32 \
    -hid_dim 256 \
    -lr 1e-5 \
    -epochs 15 \
    -device cuda:0