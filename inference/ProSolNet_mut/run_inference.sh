#!/bin/bash

feat_dpath=../../generate_features/ProSolNet_mut/outputs

python predict.py \
    -mutation_fpath ${feat_dpath}/../samples/mutation_list \
    -struc_fpath ${feat_dpath}/protein_3d_info.npy \
    -surf_fpath ${feat_dpath}/protein_surface_feats.npy \
    -lm_fpath ${feat_dpath}/protein_prott5_embedding.npy \
    -model_dpath saved_models \
    -out_fpath solubility_mut_prediction.csv \
    -batch_size 64 \
    -hid_dim 256 \
    -device cuda:0