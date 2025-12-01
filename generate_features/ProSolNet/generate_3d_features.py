import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import numpy as np
from src.structure.parse_pdb import Protein

def parse_pdb_structure(inp_dpath, out_dpath):
    all_pdb_files = [x for x in os.listdir(inp_dpath) if x.endswith("pdb")]
    
    all_struc_dict = {}
    seq_writer = open(f"{out_dpath}/protein_seqs.fasta", "w")
    for pdb in all_pdb_files:
        protein = Protein(f"{inp_dpath}/{pdb}")
        prot_info_dict = protein.process()
        basename = os.path.basename(pdb).split(".")[0]
        all_struc_dict[basename] = prot_info_dict
        prot_seq = "".join(prot_info_dict["res"])
        seq_writer.writelines(f">{basename}\n")
        seq_writer.writelines(f"{prot_seq}\n")
    seq_writer.close()
    np.save(f"{out_dpath}/protein_3d_info.npy", all_struc_dict)

if __name__ == "__main__":

    from pathlib import Path
    current_dpath = Path(__file__).resolve().parent

    inp_dpath = f"{current_dpath}/samples"
    out_dpath = f"{current_dpath}/outputs"
    os.makedirs(out_dpath, exist_ok=True)
    parse_pdb_structure(inp_dpath, out_dpath)