import os, sys 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from src.structure.prott5_func import get_lm_embedding
from transformers import T5EncoderModel, T5Tokenizer
from Bio import SeqIO
import numpy as np

def generate_lm_embedding(seq_fasta, tokenizer, lm_model, out_fpath, device):

    records = SeqIO.parse(seq_fasta, "fasta")
    keys = []
    seqs = []
    for record in records:
        keys.append(record.id)
        seqs.append(str(record.seq))

    final_feat_dict = {}
    for k, seq in zip(keys, seqs):
        lm_embedding = get_lm_embedding(seq, tokenizer, lm_model, device)
        final_feat_dict[k] = lm_embedding
    np.save(out_fpath, final_feat_dict)

if __name__ == "__main__":

    from pathlib import Path
    current_dpath = Path(__file__).resolve().parent
    
    device = "cuda:0"
    # Add the local paths for prot_t5_xl_uniref50
    prott5_model_dpath = "/your_path/prot_t5_xl_uniref50/"
    tokenizer = T5Tokenizer.from_pretrained(prott5_model_dpath, do_lower_case=False)
    lm_model = T5EncoderModel.from_pretrained(prott5_model_dpath).to(device)
    
    out_dpath = f"{current_dpath}/outputs"
    os.makedirs(out_dpath, exist_ok=True)
    out_fpath = f"{out_dpath}/protein_prott5_embedding.npy"
    generate_lm_embedding(seq_fasta=f"{out_dpath}/protein_seqs.fasta", 
                     tokenizer=tokenizer, 
                     lm_model=lm_model, 
                     out_fpath=out_fpath, 
                     device=device)
    