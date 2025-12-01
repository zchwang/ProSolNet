import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, Dataset
from argparse import RawDescriptionHelpFormatter
import argparse
from model import SolubilityNet
import copy

class Datasets(Dataset):
    def __init__(self, keys, feat_list, xyz_list, surf_feat_list, surf_res, lm_list,
                 wt_feat_list, wt_xyz_list, wt_surf_feat_list, wt_surf_res, wt_lm_list):
        self.keys = keys
        self.feat_list = feat_list
        self.xyz_list = xyz_list
        self.surf_feat_list = surf_feat_list
        self.surf_res = surf_res
        self.lm_list = lm_list
        self.wt_feat_list = wt_feat_list
        self.wt_xyz_list = wt_xyz_list
        self.wt_surf_feat_list = wt_surf_feat_list
        self.wt_surf_res = wt_surf_res
        self.wt_lm_list = wt_lm_list

    def __getitem__(self, idx):
        return self.keys[idx], self.feat_list[idx], self.xyz_list[idx], \
            self.surf_feat_list[idx], self.surf_res[idx], self.lm_list[idx], self.wt_feat_list[idx], \
            self.wt_xyz_list[idx], self.wt_surf_feat_list[idx], self.wt_surf_res[idx], self.wt_lm_list[idx]

    def __len__(self):
        return len(self.keys)
    
    def _get_data(self, lm_feats, feats, xyz, surf_feat, surf_res):
        counts = 0
        final_surf_feat = []
        final_surf_res = []
        res_batch = []
        for num, (surf_f, surf_r, feat) in enumerate(zip(surf_feat, surf_res, feats)):
            final_surf_feat.append(surf_f)
            surf_r += counts
            final_surf_res.append(surf_r)
            counts += len(feat)
            res_batch += [num] * len(th.unique(surf_r))
        final_surf_feat = th.cat(final_surf_feat, axis=0).to(th.float32)
        final_surf_res = th.cat(final_surf_res, axis=0).to(th.int64)
        res_batch = th.tensor(res_batch).to(th.int64)

        batch = []
        for num, feat in enumerate(feats):
            batch += [num] * len(feat)
        batch = th.tensor(batch).to(th.int64)
        feats = th.cat(feats, axis=0)
        lm_feats = th.cat(lm_feats, axis=0)
        xyz_tensor = th.cat(xyz, axis=0)

        return lm_feats, feats, xyz_tensor, final_surf_feat, final_surf_res, res_batch, batch

    def data_collate(self, data):
        keys, feats, xyz, surf_feat, surf_res, lm_feats, wt_feats, wt_xyz, wt_surf_feat, wt_surf_res, wt_lm_feats = zip(*data)

        lm_feats, feats, xyz_tensor, final_surf_feat, final_surf_res, res_batch, batch = \
            self._get_data(lm_feats, feats, xyz, surf_feat, surf_res)
        wt_lm_feats, wt_feats, wt_xyz_tensor, wt_final_surf_feat, wt_final_surf_res, wt_res_batch, wt_batch = \
            self._get_data(wt_lm_feats, wt_feats, wt_xyz, wt_surf_feat, wt_surf_res)

        return {"keys": keys, "feats": feats, "xyz": xyz_tensor, "surf_feats": final_surf_feat,
                "surf_res": final_surf_res, "lm_feat": lm_feats, "res_batch": res_batch, "batch": batch, 
                "wt_feats": wt_feats, "wt_xyz": wt_xyz_tensor, "wt_surf_feats": wt_final_surf_feat, 
                "wt_surf_res": wt_final_surf_res, "wt_lm_feat": wt_lm_feats, "wt_res_batch": wt_res_batch,
                "wt_batch": wt_batch}

class ProteinDataset():
    def __init__(self, mutation_fpath, struc_fpath, surf_fpath, lm_fpath, batch_size):
        self.mutation_fpath = mutation_fpath
        self.batch_size = batch_size
        self.struc_dict = np.load(struc_fpath, allow_pickle=True).item()
        self.surf_dict = np.load(surf_fpath, allow_pickle=True).item()
        self.lm_dict = np.load(lm_fpath, allow_pickle=True).item()

        self.aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
        self.dssp_list = ["H", "B", "E", "G", "I", "T", "S", " "]
    
    def get_one_hot(self, c, choice):
        _tensor = th.zeros(len(choice))
        index = choice.index(c)
        _tensor[index] = 1
        return _tensor

    def get_res_index(self, c, choice):
        index = choice.index(c)
        return index

    def get_phy_chem_feat(self, res):
        if res in ["G", "A", "V", "L", "I", "M", "F", "P", "W", "C"]:
            polar_feat = th.tensor([1, 0, 0, 0])
        elif res in ["N", "Q", "S", "T", "Y"]:
            polar_feat = th.tensor([0, 1, 0, 0])
        elif res in ["R", "K", "H"]:
            polar_feat = th.tensor([0, 0, 1, 0])
        elif res in ["D", "E"]:
            polar_feat = th.tensor([0, 0, 0, 1])
        else:
            polar_feat = th.tensor([0, 0, 0, 0])

        if res in ["F", "Y", "W"]:
            aro_feat = th.tensor([1, 0])
        else:
            aro_feat = th.tensor([0, 1])

        if res in ["A", "V", "L", "I", "F", "M", "W", "P"]:
            hydro = th.tensor([1, 0, 0])
        elif res in ["S", "T", "Y", "N", "Q", "C", "D", "E", "K", "R", "H"]:
            hydro = th.tensor([0, 0, 1])
        elif res == "G":
            hydro = th.tensor([0, 1, 0])
        else:
            hydro = th.tensor([0, 0, 0])

        feat = th.cat([polar_feat, aro_feat, hydro], axis=0)
        return feat

    def process_feats(self, keys, shuffle):
        
        res_feat_list = []
        surf_feat_list = []
        surf_res_index = []
        lm_feat_list = []
        xyz_list = []

        wt_res_feat_list = []
        wt_surf_feat_list = []
        wt_surf_res_index = []
        wt_lm_feat_list = []
        wt_xyz_list = []

        final_keys = []
        for key in keys:
            if key in list(self.struc_dict.keys()) and key in list(self.surf_dict.keys()):
                
                d = self.struc_dict[key]
                resid = d["res"]
                xyz = d["xyz"]
                dssp = d["dssp"]
                if len(xyz) > 1000:
                    continue
                final_keys.append(key)
                surf_d = self.surf_dict[key]
                surf_feat_list.append(th.from_numpy(surf_d["feat"]).to(th.float32))
                surf_res_index.append(th.from_numpy(surf_d["res_idx"]).to(th.float32))
                lm_feat_list.append(th.from_numpy(self.lm_dict[key]).to(th.float32))
                
                res_one_hot = th.cat([self.get_one_hot(res, self.aa_list).reshape(1, -1) for res in resid], axis=0).to(th.float32)
                dssp_one_hot = th.cat([self.get_one_hot(d, self.dssp_list).reshape(1, -1) for d in dssp], axis=0).to(th.float32)
                phy_chem_feat = th.cat([self.get_phy_chem_feat(i).reshape(1, -1) for i in resid], axis=0)
                res_feat_list.append(th.cat([res_one_hot, phy_chem_feat, dssp_one_hot], axis=-1))
                xyz_list.append(th.from_numpy(xyz).to(th.float32))

                wt_k = key.split("-")[0]
                wt_d = self.struc_dict[wt_k]
                wt_surf_d = self.surf_dict[wt_k]
                wt_surf_feat_list.append(th.from_numpy(wt_surf_d["feat"]).to(th.float32))
                wt_surf_res_index.append(th.from_numpy(wt_surf_d["res_idx"]).to(th.float32))
                wt_lm_feat_list.append(th.from_numpy(self.lm_dict[wt_k]).to(th.float32))
                wt_resid = wt_d["res"]
                wt_xyz = wt_d["xyz"]
                wt_dssp = wt_d["dssp"]
                wt_res_one_hot = th.cat([self.get_one_hot(res, self.aa_list).reshape(1, -1) for res in wt_resid], axis=0).to(th.float32)
                wt_dssp_one_hot = th.cat([self.get_one_hot(_d, self.dssp_list).reshape(1, -1) for _d in wt_dssp], axis=0).to(th.float32)
                wt_phy_chem_feat = th.cat([self.get_phy_chem_feat(i).reshape(1, -1) for i in wt_resid], axis=0)
                wt_res_feat_list.append(th.cat([wt_res_one_hot, wt_phy_chem_feat, wt_dssp_one_hot], axis=-1))
                wt_xyz_list.append(th.from_numpy(wt_xyz).to(th.float32))

        dataset = Datasets(final_keys, res_feat_list, xyz_list, surf_feat_list, surf_res_index, lm_feat_list,
                           wt_res_feat_list, wt_xyz_list, wt_surf_feat_list, wt_surf_res_index, wt_lm_feat_list)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=dataset.data_collate, num_workers=2)

        return final_keys, dataloader
    
    def get_dataloader(self):
        with open(self.mutation_fpath) as f:
            keys = [x.strip() for x in f.readlines()]
        final_keys, dataloader = self.process_feats(keys, False)

        return final_keys, dataloader

def recover_pred(pred_tensor):
    interval = 2/3
    pred_1 = (pred_tensor >= -1.) * (pred_tensor < interval-1) * -1
    pred_2 = (pred_tensor >= interval-1) * (pred_tensor < 2*interval-1) * 0
    pred_3 = (pred_tensor >= 2*interval-1) * (pred_tensor <= 1) * 1
    return (pred_1 + pred_2 + pred_3)

def run_an_eval_epoch(model, dataloader, device):
    model.eval()
    with th.no_grad():
        all_keys = []
        pred_list = []
        for step, data in enumerate(dataloader):
            keys, feats, xyz, surf_feats, surf_res, lm_feats, res_batch, batch, wt_feats, \
                wt_xyz, wt_surf_feats, wt_surf_res, wt_lm_feats, wt_res_batch, wt_batch = list(data.values())
            all_keys += keys
            feats = feats.to(device)
            xyz = xyz.to(device)
            surf_feats = surf_feats.to(device)
            surf_res = surf_res.to(device)
            lm_feats = lm_feats.to(device)
            res_batch = res_batch.to(device)
            batch = batch.to(device)
            wt_feats = wt_feats.to(device)
            wt_xyz = wt_xyz.to(device)
            wt_surf_feats = wt_surf_feats.to(device)
            wt_surf_res = wt_surf_res.to(device)
            wt_lm_feats = wt_lm_feats.to(device)
            wt_res_batch = wt_res_batch.to(device)
            wt_batch = wt_batch.to(device)
            _, prediction = model(feats, xyz, surf_feats, surf_res, lm_feats, res_batch, batch,
                              wt_feats, wt_xyz, wt_surf_feats, wt_surf_res, wt_lm_feats, wt_res_batch, wt_batch)
           
            pred_list.append(prediction.cpu().detach().numpy().ravel())
        
        N = len(dataloader)
        pred_array = np.concatenate(pred_list, axis=0)

        return pred_array


if __name__ == "__main__":
    d = "Training ..."
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-mutation_fpath", type=str, default="mutation_list",
                         help="Input. The mutation list.")
    parser.add_argument("-struc_fpath", type=str, default="protein_3d_info.npy",
                         help="Input. The file path for the 3d strctural information of proteins.")
    parser.add_argument("-surf_fpath", type=str, default="protein_surface_feats.npy",
                         help="Input. The file path for the surface features of proteins.")
    parser.add_argument("-lm_fpath", type=str, default="protein_prott5_embedding.npy",
                         help="Input. The file path of prott5 embedding.")
    parser.add_argument("-model_dpath", type=str, default="saved_model.pth",
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
    mutation_fpath = args.mutation_fpath
    struc_fpath = args.struc_fpath
    surf_fpath = args.surf_fpath
    lm_fpath = args.lm_fpath
    model_dpath = args.model_dpath
    out_fpath = args.out_fpath
    batch_size = args.batch_size 
    hid_dim = args.hid_dim
    device = args.device 
    dataset = ProteinDataset(mutation_fpath, struc_fpath, surf_fpath, lm_fpath, batch_size)
    test_keys, test_dataloader = dataset.get_dataloader()
    
    all_preds = []
    for fold in range(1, 11):
        model = SolubilityNet(hid_dim=hid_dim).to(device)
        model.load_state_dict(th.load(f"{model_dpath}/model-{fold}.pth", map_location=device))
        pred_array = run_an_eval_epoch(model, test_dataloader, device)
        all_preds.append(pred_array.reshape(-1, 1))
    
    all_preds = np.concatenate(all_preds, axis=-1)
    mean_pred = np.mean(all_preds, axis=-1)
    final_pred_class = recover_pred(mean_pred)
    final_all_preds = np.concatenate([all_preds, mean_pred.reshape(-1, 1), final_pred_class.reshape(-1, 1)], axis=-1)
    
    columns = [f"model-{fold}" for fold in range(1, 11)] + ["mean", "pred_class"]
    final_df = pd.DataFrame(final_all_preds, index=test_keys, columns=columns)

    final_df.to_csv(out_fpath)