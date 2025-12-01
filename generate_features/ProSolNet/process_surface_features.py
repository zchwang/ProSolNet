import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from src.surface.parse_ply import ProteinSurface
import numpy as np

def process_ply(pdb_dpath, ply_dpath, out_fpath):
    all_ply = [x for x in os.listdir(ply_dpath) if x.endswith("ply")]

    all_surf_dict = {}
    for ply in all_ply:
        basename = os.path.basename(ply).split(".")[0]
        pdb_fpath = f"{pdb_dpath}/{basename}.pdb"
        ply_fpath = f"{ply_dpath}/{basename}.ply"
        prot = ProteinSurface(pdb_fpath, ply_fpath)
        prot.parse_pdb()
        feat_dict = prot.parse_ply()
        all_surf_dict[basename] = feat_dict 
    
    np.save(out_fpath, all_surf_dict)

if __name__ == "__main__":

    from pathlib import Path
    current_dpath = Path(__file__).resolve().parent

    pdb_dpath = f"{current_dpath}/samples"
    ply_dpath = f"{current_dpath}/outputs/ply_outputs"    
    out_fpath = f"{current_dpath}//outputs/protein_surface_feats.npy"

    process_ply(pdb_dpath, ply_dpath, out_fpath)