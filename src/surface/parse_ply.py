import mdtraj as md
import numpy as np
from scipy.spatial.distance import cdist

class ProteinSurface():
    def __init__(self, pdb_fpath, ply_fpath):
        self.ply_fpath = ply_fpath
        self.pdb = md.load_pdb(pdb_fpath)
        self.top = self.pdb.topology
        self.res_3_to_1_dict = {"GLY": "G", "ALA": "A", "VAL": "V", "LEU": "L", "ILE": "I", "PHE": "F", "PRO": "P",
                               "SER": "S", "THR": "T", "HIS": "H", "TRP": "W", "CYS": "C", "ASP": "D", "GLU": "E",
                               "LYS": "K", "TYR": "Y", "MET": "M", "ASN": "N", "GLN": "Q", "ARG": "R"}
    def parse_pdb(self):
        all_res_xyz = []
        self.all_res_names = []
        for num, residue in enumerate(self.top.residues):
            _res_xyz = []
            res_name = residue.code
            self.all_res_names.append(res_name)
            for atom in residue.atoms:
                index = atom.index
                _xyz = [float(x) for x in self.pdb.xyz[0, index] * 10]
                _res_xyz.append(_xyz)
            _res_xyz = np.array(_res_xyz)
            
            all_res_xyz.append(_res_xyz)
  
        self.all_res_xyz = all_res_xyz.copy()
        return self
    
    def parse_ply(self):
        with open(self.ply_fpath) as f:
            lines = [x for x in f.readlines() if len(x.split()) == 10]

        mesh_coords = []
        feat_list = []
        for line in lines:
            x = float(line.split()[0])
            y = float(line.split()[1])
            z = float(line.split()[2])
            charge = float(line.split()[3])
            hbond = float(line.split()[4])
            hphob = float(line.split()[5])
            mesh_coords.append([x, y, z])
            feat_list.append([charge, hbond, hphob])
        
        mesh_coords = np.array(mesh_coords)
        feat_array = np.array(feat_list)

        all_min_dist = []
        for res_xyz in self.all_res_xyz:
            _d = cdist(mesh_coords, res_xyz, metric='euclidean')
            _min_d = _d.min(1)
            all_min_dist.append(_min_d)
        all_min_dist = np.array(all_min_dist)
        all_min_index = np.argmin(all_min_dist, axis=0)
        feat_dict = {"feat": feat_array, "res_idx": all_min_index}

        return feat_dict
