from pyrosetta import *
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
#from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.protocols.relax import FastRelax
import os

init()

aa_dict = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'E': 'GLU',
    'Q': 'GLN',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL'
}

def mutate(pdb, mutation, out_dpath):
    if os.path.exists(pdb):
        basename = os.path.basename(pdb).split(".")[0]
        pose = pose_from_pdb(pdb)
        muts_list = mutation.split(",")
        mut_str = "-".join(muts_list)
        switch = True
        for mut in muts_list:
            src = mut[0]
            dst = mut[-1]
            site = int(mut[1:-1])
            cur_res = pose.residue(site).name()
            if cur_res != aa_dict[src]:
                switch = False
                break
            else:
                MutateResidue(site, aa_dict[dst]).apply(pose)
        if switch:
            relax = FastRelax()
            relax.set_scorefxn(get_fa_scorefxn())
            pose.dump_pdb(f"{out_dpath}/{basename}-{mut_str}.pdb")

if __name__ == "__main__":

    with open("mutation_list") as f:
        mutations = [x.strip() for x in f.readlines()]

    mut_output_dpath = "."
    os.makedirs(mut_output_dpath, exist_ok=True)
    for mutant in mutations:
        code, mut = mutant.split("-")
        pdb_fpath = f"{code}.pdb"
        mutate(pdb_fpath, mut, mut_output_dpath)