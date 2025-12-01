import os
import subprocess
import random
import shutil

# To generate protein surface features using MaSIF, you need to install the required environment dependencies 
# according to MaSIF's specifications. Then, add the following dependencies to the environment path.
os.environ["APBS_BIN"] = "/your_path/APBS-3.0.0.Linux/bin/apbs"
os.environ["LD_LIBRARY_PATH"] = f"/your_path/APBS-3.0.0.Linux/lib:{os.getenv('LD_LIBRARY_PATH', '')}"
os.environ["MULTIVALUE_BIN"] = "/your_path/APBS-3.0.0.Linux/share/apbs/tools/bin/multivalue"
os.environ["PDB2PQR_BIN"] = "/your_path/pdb2pqr-linux-bin64-2.1.1/pdb2pqr"
os.environ["REDUCE_HET_DICT"] = "/your_path/reduce-4.12/reduce_wwPDB_het_dict.txt"
os.environ["PYMESH_PATH"] = "/your_path/PyMesh-0.3"
os.environ["MSMS_BIN"] = "/your_path/msms/msms.x86_64Linux2.2.6.1.staticgcc"
os.environ["PDB2XYZRN"] = "/your_path/msms/pdb_to_xyzrn"

def get_masif_feats(pdb_dpath, ply_dpath, tmp_dir): 
    all_pdb = [x for x in os.listdir(pdb_dpath) if x.endswith(".pdb")]
    os.chdir(tmp_dir)
    for pdb in all_pdb:
        basename = os.path.basename(pdb).split(".")[0]
        seed = str(random.randint(0, 1000000000))
        os.makedirs(f"global_{seed}")
        os.chdir(f"global_{seed}")
        cmd = f"bash {script_path}/data_prepare_one.sh --file {pdb_dpath}/{basename}.pdb {basename}_A"
        subprocess.run(cmd, shell=True, check=True)
        shutil.copyfile(f"{tmp_dir}/global_{seed}/data_preparation/01-benchmark_surfaces/{basename}_A.ply",
                        f"{ply_dpath}/{basename}.ply")
        os.chdir("../")
        shutil.rmtree(f"global_{seed}")
        
if __name__ == "__main__":

    from pathlib import Path
    current_dpath = Path(__file__).resolve().parent

    script_path = f"{current_dpath}/../../src/surface"
    inp_dpath = f"{current_dpath}/samples"
    ply_dpath = f"{current_dpath}/outputs/ply_outputs"
    tmp_dir = f"{current_dpath}/outputs/tmp"
    os.makedirs(ply_dpath, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    get_masif_feats(inp_dpath, ply_dpath, tmp_dir)