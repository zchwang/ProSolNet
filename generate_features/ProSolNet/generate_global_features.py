import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from src.sequence.seq_func import SequenceFeatures
from Bio import SeqIO

def get_global_feats(fasta_fpath, training_ref_fpath, out_fpath):
    
    training_ref_df = pd.read_csv(training_ref_fpath, index_col=0)
    mean = training_ref_df.values[0].reshape(1, -1)
    std = training_ref_df.values[1].reshape(1, -1)

    records = SeqIO.parse(fasta_fpath, "fasta")
    keys = []
    feats = []
    for record in records:
        seq = str(record.seq)
        sequence = SequenceFeatures(seq=seq, userch_bin=userch_bin, tmhmm_bin=tmhmm_bin, tmp_dir=tmp_dir)
        seq_feats = sequence.get_seq_feats()
        feats.append(seq_feats.reshape(1, -1))
        keys.append(record.id)
    
    feats = np.concatenate(feats, axis=0)
    norm_feats = (feats - mean) / std 

    feat_df = pd.DataFrame(norm_feats, index=keys)
    feat_df.to_pickle(out_fpath)
    
if __name__ == "__main__":

    from pathlib import Path
    current_dpath = Path(__file__).resolve().parent

    # Add the local paths for TMHMM and USEARCH
    tmhmm_bin = "/your_path/tmhmm/tmhmm-2.0c/bin"
    userch_bin = "/your_path/usearch"
    
    training_ref_fpath = f"{current_dpath}/../../src/sequence/training-set_seq-mean-std.csv"
    out_dpath = f"{current_dpath}/outputs"
    tmp_dir = f"{current_dpath}/{out_dpath}/tmp"

    os.makedirs(out_dpath, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    get_global_feats(
        fasta_fpath=f"{out_dpath}/protein_seqs.fasta",
        training_ref_fpath=training_ref_fpath,
        out_fpath=f"{out_dpath}/protein_global_feats.pkl"
    )

    

