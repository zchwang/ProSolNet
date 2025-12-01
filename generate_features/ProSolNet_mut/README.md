The Python scripts in this directory are used to generate the features required by ProSolNet_mut. Before running them, we need to configure the relevant environment paths inside the scripts.

## Usage

### 1. Generate the mutant structures from the wild-type protein. Generate mutants based on the wild-type protein PDB file and the mutation list. In the `samples` directory, run

        python generate_mutants.py

### 2. Load the 3D structure of the protein.

        python generate_3d_features.py

### 3. Generate residue-level protein features using the `Prot_t5_xl_uniref50` language model. Specify the local path of `Prot_t5_xl_uniref50` in the `generate_prott5_embedding.py` script. Then run:
        
        python generate_prott5_embedding.py

### 4. Generate protein surface features. We need to first install [MaSIF](https://github.com/LPDI-EPFL/masif). The environment required by the MaSIF differs from that used in the ProSolNet environment. Therefore, MaSIF needs to be installed in a separate environment. Under the MaSIF environment, run the following command:
        
        python generate_surface_features.py

This step will generate `.ply` files.

### 5. Process `.ply` files under the ProSolNet environment.
        
        python process_surface_features.py


After completing the five steps above, all the required features for ProSolNet will be generated.