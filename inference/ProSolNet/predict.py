import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, Dataset
from argparse import RawDescriptionHelpFormatter
import argparse
from model import SolubilityNet
import copy

class Datasets(Dataset):
    def __init__(self, keys, seq_feat, feat_list, xyz_list, surf_feat_list, surf_res):
        self.keys = keys
        self.seq_feat = seq_feat
        self.feat_list = feat_list
        self.xyz_list = xyz_list
        self.surf_feat_list = surf_feat_list
        self.surf_res = surf_res

    def __getitem__(self, idx):
        return self.keys[idx], self.seq_feat[idx].unsqueeze(0), self.feat_list[idx], self.xyz_list[idx], \
            self.surf_feat_list[idx], self.surf_res[idx]

    def __len__(self):
        return len(self.keys)
    
    def data_collate(self, data):
        keys, seq_feats, feats, xyz, surf_feat, surf_res = zip(*data)

        counts = 0
        final_surf_feat = []
        final_surf_res = []
        res_batch = []
        batch = []
        for num, (surf_f, surf_r, feat) in enumerate(zip(surf_feat, surf_res, feats)):
            final_surf_feat.append(surf_f)
            surf_r += counts
            final_surf_res.append(surf_r)
            counts += len(feat)
            res_batch += [num] * len(th.unique(surf_r))
            batch += [num] * len(feat)
            
        final_surf_feat = th.cat(final_surf_feat, axis=0).to(th.float32)
        final_surf_res = th.cat(final_surf_res, axis=0).to(th.int64)
        res_batch = th.tensor(res_batch).to(th.int64)

        batch = th.tensor(batch).to(th.int64)
        seq_feats = th.cat(seq_feats, axis=0)
        feats = th.cat(feats, axis=0)
        xyz_tensor = th.cat(xyz, axis=0)

        return {"keys": keys, "seq_feats": seq_feats, "feats": feats, "xyz": xyz_tensor,
                "surf_feats": final_surf_feat, "surf_res": final_surf_res, "res_batch": res_batch, "batch": batch}

class ProteinDataset():
    def __init__(self, seq_fpath, struc_fpath, lm_fpath, surf_fpath, batch_size):
        self.batch_size = batch_size
        seq_df = pd.read_pickle(seq_fpath)
        self.seq_dict = dict(zip(seq_df.index.tolist(), seq_df.values))
        self.struc_dict = np.load(struc_fpath, allow_pickle=True).item()
        self.surf_dict = np.load(surf_fpath, allow_pickle=True).item()
        self.lm_dict = np.load(lm_fpath, allow_pickle=True).item()

    def process_feat(self, keys, shuffle):
        seq_feat_list = []
        res_feat_list = []
        surf_feat_list = []
        surf_res_index = []
        xyz_list = []
        final_keys = []
        for key in keys:
            if key in list(self.struc_dict.keys()) and key in list(self.surf_dict.keys()):
                d = self.struc_dict[key]
                final_keys.append(key)
                surf_d = self.surf_dict[key]
                surf_feat_list.append(th.from_numpy(surf_d["feat"]).to(th.float32))
                surf_res_index.append(th.from_numpy(surf_d["res_idx"]).to(th.float32))
                seq_feat = th.from_numpy(self.seq_dict[key])
                seq_feat_list.append(seq_feat.reshape(1, -1))

                resid, xyz = list(d.values())
                lm_feat = th.from_numpy(self.lm_dict[key]).to(th.float32)
                res_feat_list.append(lm_feat)
                xyz_list.append(th.from_numpy(xyz).to(th.float32))
        
        seq_feat = th.cat(seq_feat_list, axis=0).to(th.float32)
        dataset = Datasets(final_keys, seq_feat, res_feat_list, xyz_list, surf_feat_list, surf_res_index)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=dataset.data_collate, num_workers=4)

        return final_keys, dataloader
    
    def get_dataloader(self):
        test_keys = list(self.seq_dict.keys())
        final_test_keys, test_dataloader = self.process_feat(test_keys, False)
        print("test number:", len(final_test_keys))

        return final_test_keys, test_dataloader
    
def run_an_eval_epoch(model, dataloader, device):
    model.eval()
    with th.no_grad():
        final_keys = []
        all_pred_list = []
        for step, data in enumerate(dataloader):
            keys, seq_feats, feats, xyz, surf_feats, surf_res, res_batch, batch = list(data.values())
            seq_feats = seq_feats.to(device)
            feats = feats.to(device)
            xyz = xyz.to(device)
            surf_feats = surf_feats.to(device)
            surf_res = surf_res.to(device)
            res_batch = res_batch.to(device)
            batch = batch.to(device)
            _, pred = model(seq_feats, feats, xyz, surf_feats, surf_res, res_batch, batch)
        
            all_pred_list.append(pred.cpu().detach().numpy().ravel())
            final_keys += keys

        all_pred_array = np.concatenate(all_pred_list, axis=0)
        score_df = pd.DataFrame(all_pred_array, index=final_keys, columns=["pred_probability"])

        return score_df
    

if __name__ == "__main__":
    
    d = "Predict protein solubility using ProSolNet ..."

    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-global_fpath", type=str, default="protein_global_feats.pkl",
                         help="Input. The file path for the global features of proteins.")
    parser.add_argument("-struc_fpath", type=str, default="protein_3d_info.npy",
                         help="Input. The file path for the 3d strctural information of proteins.")
    parser.add_argument("-surf_fpath", type=str, default="protein_surface_feats.npy",
                         help="Input. The file path for the surface features of proteins.")
    parser.add_argument("-lm_fpath", type=str, default="protein_prott5_embedding.npy",
                         help="Input. The file path of prott5 embedding.")
    parser.add_argument("-model_fpath", type=str, default="saved_model.pth",
                         help="Input. The path of saved model.")
    parser.add_argument("-batch_size", type=int, default=64,
                        help="Input. Batch size")
    parser.add_argument("-hid_dim", type=int, default=64,
                        help="Input. The hidden embedding dimension.")
    parser.add_argument("-out_fpath", type=str, default="cuda",
                        help="Input. The path of output file.")
    parser.add_argument("-device", type=str, default="cuda:0",
                        help="Input. The device: cuda or cpu.")
    args = parser.parse_args()

    global_fpath = args.global_fpath
    struc_fpath = args.struc_fpath
    surf_fpath = args.surf_fpath
    lm_fpath = args.lm_fpath
    model_fpath = args.model_fpath
    out_fpath = args.out_fpath
    batch_size = args.batch_size
    hid_dim = args.hid_dim
    device = args.device
    dataset = ProteinDataset(global_fpath, struc_fpath, lm_fpath, surf_fpath, batch_size)
    test_keys, test_dataloader = dataset.get_dataloader()

    model = SolubilityNet(hid_dim=hid_dim).to(device)
    model.load_state_dict(th.load(model_fpath, map_location=device))
    t_score_df = run_an_eval_epoch(model, test_dataloader, device)
    t_score_df.to_csv(out_fpath)
    