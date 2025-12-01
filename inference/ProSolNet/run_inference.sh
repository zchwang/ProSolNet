#!/bin/bash

feat_dpath=../../generate_features/ProSolNet/outputs
device=cuda:0

python predict.py \
    -global_fpath ${feat_dpath}/protein_global_feats.pkl \
    -struc_fpath ${feat_dpath}/protein_3d_info.npy \
    -surf_fpath ${feat_dpath}/protein_surface_feats.npy \
    -lm_fpath ${feat_dpath}/protein_prott5_embedding.npy \
    -model_fpath saved_model.pth \
    -batch_size 64 \
    -hid_dim 64 \
    -out_fpath solubility_prediction.csv \
    -device $device