The Python scripts in this directory are used to generate the features required by ProSolNet. Before running them, we need to configure the relevant environment paths inside the scripts.

## Usage

Here, we generate features for the proteins in the `./samples` directory by following the steps below.

1. Load the 3D structure of the protein.

        python generate_3d_features.py

2. Generate residue-level protein features using the `Prot_t5_xl_uniref50` language model. Specify the local path of `Prot_t5_xl_uniref50` in the `generate_prott5_embedding.py` script. Then run:
        
        python generate_prott5_embedding.py

3. Generate global protein features. Install `TMHMM` and `USEARCH`, and add their installation paths to the `generate_global_features.py` script. Then run:
        
        python generate_global_features.py

4. Generate protein surface features. We need to first install [MaSIF](https://github.com/LPDI-EPFL/masif). The environment required by the MaSIF differs from that used in the ProSolNet environment. Therefore, MaSIF needs to be installed in a separate environment. Under the MaSIF environment, run the following command:
        
        python generate_surface_features.py

This step will generate `.ply` files.

5. Process `.ply` files under the ProSolNet environment.
        
        python process_surface_features.py


After completing the five steps above, all the required features for ProSolNet will be generated.