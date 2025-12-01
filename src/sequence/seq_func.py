import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import molecular_weight
from collections import Counter
from itertools import combinations
import numpy as np
import os
import random

class SequenceFeatures():
    def __init__(self, seq, userch_bin, tmhmm_bin, tmp_dir="./tmp"):
        self.userch_bin = userch_bin
        self.tmhmm_bin = tmhmm_bin

        self.seq = seq
        self.process_seq = seq.replace("X", "A")
        self.tmp_dir = tmp_dir
        self.random_name = str(random.randint(0, 10000000))
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.analyzed_seq = ProteinAnalysis(self.seq)
        self.analyzed_process_seq = ProteinAnalysis(self.process_seq)
        self.aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', "X"]

    def cal_total_dimers(self):
        self_pairs = [f"{a}{a}" for a in self.aa_list]
        other_pairs = ["".join(sorted(x)) for x in list(combinations(self.aa_list, 2))]
        self.all_dimers = self_pairs + other_pairs
        return self

    def cal_dimer_percent(self):
        self.cal_total_dimers()
        dimers = ["".join(sorted(self.seq[i:i+2])) for i in range(len(self.seq) - 1)]
        dimer_dict = {}.fromkeys(self.all_dimers, 0)
        for d in dimers:
            dimer_dict[d] += 1
        dimer_percent = np.array(list(dimer_dict.values()))/len(dimers)
        return dimer_percent.tolist()

    def get_seq_content(self):
        aa_count_dict = {}.fromkeys(self.aa_list, 0)
        for a in self.seq:
            aa_count_dict[a] += 1
        aa_counts = np.array(list(aa_count_dict.values()))
        aa_frac = (aa_counts / len(self.seq)).tolist()
        dimer_percent = self.cal_dimer_percent()
        return aa_frac, dimer_percent
    
    def cal_charge_percent(self):
        chrg_counts = 0
        for a in self.seq:
            if a in ['K', 'R', 'H', 'D', 'E']:
                chrg_counts += 1
        chrg_percent = chrg_counts/len(self.seq)
        return chrg_percent
    
    def cal_physicochemical_feats(self):
        process_seqs = self.seq.replace("X", "A")
        weight = molecular_weight(process_seqs, seq_type="protein")
        chrg_percent = self.cal_charge_percent()
        pi = self.analyzed_seq.isoelectric_point()
        aromatic = self.analyzed_seq.aromaticity()
        instability = self.analyzed_process_seq.instability_index()
        gravy = self.analyzed_process_seq.gravy()
        ss = list(self.analyzed_seq.secondary_structure_fraction()) # 3 ["helix", "sheet", "coil"]
        charges = []
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            charges.append(self.analyzed_seq.charge_at_pH(i))
        return [weight, chrg_percent, pi, aromatic, instability, gravy] + ss + charges 
    
    def tmhmm_userch_feats(self):
        seq_record = [SeqRecord(Seq(self.seq), id=self.random_name, description="")]
        with open(f"{self.tmp_dir}/{self.random_name}.fasta", "w") as f:
            SeqIO.write(seq_record, f, "fasta")
      
        tmhmm_cmd = f"/usr/bin/perl {self.tmhmm_bin}/tmhmm {self.tmp_dir}/{self.random_name}.fasta >> {self.tmp_dir}/{self.random_name}_tmhmm_out.dat"
        os.system(tmhmm_cmd)
        with open(f"{self.tmp_dir}/{self.random_name}_tmhmm_out.dat") as f:
            lines = [x.strip() for x in f.readlines()]
        AAs_TMHs = None
        first_60_AAs = None 
        prob_N_in = None
        for line in lines:
            if not line.startswith("#"):
                continue
            if "Length:" in line or "Number of predicted TMHs:" in line:
                continue
            if "Exp number of AAs in TMHs" in line:
                AAs_TMHs = float(line.split()[-1])
            elif "Exp number, first 60 AAs" in line:
                first_60_AAs = float(line.split()[-1])
            elif "Total prob of N-in" in line:
                prob_N_in = float(line.split()[-1])
        tmhmm_feats = [AAs_TMHs, first_60_AAs, prob_N_in]
        
        userch_cmd = f"{self.userch_bin}/usearch -usearch_global {self.tmp_dir}/{self.random_name}.fasta -db {self.userch_bin}/Ecoli_xray_nmr_pdb_no_nesg_simple_id.fasta -id 0.01 -blast6out {self.tmp_dir}/{self.random_name}_userch_out.dat"
        os.system(userch_cmd)
        with open(f"{self.tmp_dir}/{self.random_name}_userch_out.dat") as f:
            line = f.readline()
        
        if len(line) == 0:
            identity = 0.0
        else:
            identity = float(line.split("\t")[2].strip()) / 100
        os.remove(f"{self.tmp_dir}/{self.random_name}_userch_out.dat")
        os.remove(f"{self.tmp_dir}/{self.random_name}_tmhmm_out.dat")
        os.remove(f"{self.tmp_dir}/{self.random_name}.fasta")
        #os.rmdir(f"TMHMM_{self.random_name}")
        os.system(f"rm -rf TMHMM*")

        return [identity] + tmhmm_feats
    
    def get_columns(self):
        aa_col = [f"frac_{a}" for a in self.aa_list]
        dimer_col = [f"frac_{dimer}" for dimer in self.all_dimers]
        physchem_col = ["mol_weight", "chr_frac", "pi", "aromatic_frac", "instability", "gravy", "frac_helix", "frac_sheet", "frac_coil"] + [f"chr_ph{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
        tmhmm_userch_col = ["identity", "AAs_TMHs" ,"first 60 AAs", "N_in_cytop"]
        all_col = aa_col + dimer_col + physchem_col + tmhmm_userch_col
        return all_col

    def get_seq_feats(self):
        aa_percent, dimer_percent = self.get_seq_content()
        phychem_feats = self.cal_physicochemical_feats()
        tmhmm_userch_feats = self.tmhmm_userch_feats()
        feats = np.array(aa_percent + dimer_percent + phychem_feats + tmhmm_userch_feats)

        return feats
    
def generate_feats(dir_, key, seq):
    os.makedirs(f"{dir_}/{key}", exist_ok=True)
    os.chdir(f"{dir_}/{key}")
    if not os.path.exists(f"{key}.csv"): 
        try: 
            pro = SequenceFeatures(seq)
            all_feats = pro.get_seq_feats()
            all_col = pro.get_columns()
            df = pd.DataFrame(np.array(all_feats).reshape(1, -1), index=[key], columns=all_col)
            df.to_csv(f"{key}.csv")
        except Exception as e:
            print("Error", e, key)
    os.chdir("../../")
 