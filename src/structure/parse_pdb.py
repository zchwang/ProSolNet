import os, sys
import numpy as np
import pandas as pd
import mdtraj as md
import multiprocessing

class Protein():
    def __init__(self, fpath):
        self.fpath = fpath
        self.pdb = md.load_pdb(fpath)
        self.top = self.pdb.topology
        self.res_3_to_1_dict = {"GLY": "G", "ALA": "A", "VAL": "V", "LEU": "L", "ILE": "I", "PHE": "F", "PRO": "P",
                               "SER": "S", "THR": "T", "HIS": "H", "TRP": "W", "CYS": "C", "ASP": "D", "GLU": "E",
                               "LYS": "K", "TYR": "Y", "MET": "M", "ASN": "N", "GLN": "Q", "ARG": "R"}
                    
    def process(self, have_dssp=False):

        res_xyz = []
        self.res_names = []
        self.ha_to_res_idx_dict = {}
        for num, residue in enumerate(self.top.residues):
            res = residue.name
            try:
                self.res_names.append(self.res_3_to_1_dict[res])
            except KeyError:
                self.res_names.append("X")
            ha_xyz_dict = {}
            
            for atom in residue.atoms:
                ele = atom.element.symbol
                if ele == "H":
                    continue
                index = atom.index
                _xyz = self.pdb.xyz[0, index] * 10
                _type = atom.name
                ha_xyz_dict[_type] = _xyz
                self.ha_to_res_idx_dict[index] = num
            
            if "CA" in list(ha_xyz_dict.keys()):
                xyz = ha_xyz_dict["CA"]
            elif "CB" in list(ha_xyz_dict.keys()):
                xyz = ha_xyz_dict["CB"]
            else:
                xyz = np.mean(np.concatenate(list(ha_xyz_dict.values()), axis=0).reshape(-1, 3), axis=0)
            res_xyz.append(xyz)

        feat_dict = {"res": self.res_names, "xyz": np.array(res_xyz)}
        if have_dssp:
            dssp = md.compute_dssp(self.pdb, simplified=False).tolist()[0]
            feat_dict["dssp"] = dssp
        
        return feat_dict

