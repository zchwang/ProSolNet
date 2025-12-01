import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, Dataset
from utils import EarlyStopping, run_a_training_epoch, run_an_eval_epoch, \
        write_logfile, out_results
from argparse import RawDescriptionHelpFormatter
import argparse
from model import SolubilityNet

class Datasets(Dataset):
    def __init__(self, keys, seq_feat, feat_list, xyz_list, surf_feat_list, surf_res, label):
        self.keys = keys
        self.seq_feat = seq_feat
        self.feat_list = feat_list
        self.xyz_list = xyz_list
        self.surf_feat_list = surf_feat_list
        self.surf_res = surf_res
        self.label = label
    
    def __getitem__(self, idx):
        return self.keys[idx], self.seq_feat[idx].unsqueeze(0), self.feat_list[idx], self.xyz_list[idx], \
            self.surf_feat_list[idx], self.surf_res[idx], self.label[idx].unsqueeze(0)
    
    def __len__(self):
        return len(self.keys) 
    
    def data_collate(self, data):
        keys, seq_feats, feats, xyz, surf_feat, surf_res, labels = zip(*data)

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
        labels = th.cat(labels, axis=0)
        
        return {"keys": keys, "seq_feats": seq_feats, "feats": feats, "xyz": xyz_tensor, 
                "surf_feats": final_surf_feat, "surf_res": final_surf_res, "labels": labels, "res_batch": res_batch, "batch": batch}

class ProteinDataset():
    def __init__(self, seq_fpath, struc_fpath, lm_fpath, surf_fpath, split_fpath, label_fpath, batch_size):
        self.batch_size = batch_size
        
        seq_df = pd.read_pickle(seq_fpath)
        self.seq_dict = dict(zip(seq_df.index.tolist(), seq_df.values))
        self.struc_dict = np.load(struc_fpath, allow_pickle=True).item()
        self.lm_dict = np.load(lm_fpath, allow_pickle=True).item()
        self.surf_dict = np.load(surf_fpath, allow_pickle=True).item()  
        label_df = pd.read_csv(label_fpath, index_col=0)
        self.label_dict = dict(zip(label_df.index.tolist(), label_df.values.ravel()))
        self.split_dict = np.load(split_fpath, allow_pickle=True).item()
           
    def process_feat(self, keys, shuffle):
        seq_feat_list = []
        res_feat_list = []
        surf_feat_list = []
        surf_res_index = []
        xyz_list = []
        label_list = []
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
                label_list.append(self.label_dict[key])
        
        seq_feat = th.cat(seq_feat_list, axis=0).to(th.float32)
        label_tensor = th.from_numpy(np.array(label_list)).to(th.float32)
        dataset = Datasets(final_keys, seq_feat, res_feat_list, xyz_list, surf_feat_list, surf_res_index, label_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=dataset.data_collate, num_workers=2)

        return final_keys, dataloader

    def get_dataloader(self):

        train_keys = self.split_dict["train"]
        valid_keys = self.split_dict["valid"]

        final_train_keys, train_dataloader = self.process_feat(train_keys, True)
        final_valid_keys, valid_dataloader = self.process_feat(valid_keys, False)

        return train_dataloader, final_valid_keys, valid_dataloader
    
if __name__ == "__main__":
    
    d = "Training ProSolNet ..."

    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-global_fpath", type=str, default="protein_global_feats.pkl",
                         help="Input. The file path for the global features of proteins.")
    parser.add_argument("-struc_fpath", type=str, default="protein_3d_info.npy",
                         help="Input. The file path for the 3d strctural information of proteins.")
    parser.add_argument("-surf_fpath", type=str, default="protein_surface_feats.npy",
                         help="Input. The file path for the surface features of proteins.")
    parser.add_argument("-lm_fpath", type=str, default="protein_prott5_embedding.npy",
                         help="Input. The file path of prott5 embedding.")
    parser.add_argument("-label_fpath", type=str, default="all_label.csv",
                         help="Input. The file path of labels.")
    parser.add_argument("-split_fpath", type=str, default="data-split_dict.npy",
                         help="Input. The split of the training, validation, and test sets.")
    parser.add_argument("-batch_size", type=int, default=64,
                        help="Input. Batch size")
    parser.add_argument("-hid_dim", type=int, default=64,
                        help="Input. The hidden embedding dimension.")
    parser.add_argument("-lr", type=float, default=0.001,
                        help="Input. Learning rate.")
    parser.add_argument("-drop_rate", type=float, default=0.4,
                        help="Input. The rate of dropout.")
    parser.add_argument("-epochs", type=int, default=15,
                        help="Input. Epochs")
    parser.add_argument("-device", type=str, default="cuda",
                        help="Input. The device: cuda or cpu.")
    args = parser.parse_args()

    struc_fpath = args.struc_fpath
    global_fpath = args.global_fpath
    surf_fpath = args.surf_fpath
    label_fpath = args.label_fpath
    lm_fpath = args.lm_fpath
    split_fpath = args.split_fpath
    lr = args.lr 
    batch_size = args.batch_size 
    hid_dim = args.hid_dim
    drop_rate = args.drop_rate
    device = args.device 
    epochs = args.epochs
    patience = 5
    min_delta = 0.001

    dataset = ProteinDataset(global_fpath, struc_fpath, lm_fpath, surf_fpath, split_fpath, label_fpath, batch_size)
    train_dataloader, final_valid_keys, valid_dataloader = dataset.get_dataloader()
            
    model = SolubilityNet(hid_dim=hid_dim, drop_rate=drop_rate).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, capturable=True, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
  
    stop = EarlyStopping(patience, min_delta)
    bestmodel = None
    record_data = []
    best_valid_results = None
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        train_eval = run_a_training_epoch(model, train_dataloader, optimizer, device)
        valid_pred, valid_real, valid_eval = run_an_eval_epoch(model, valid_dataloader, device)
        record_data.append(np.concatenate([np.array([epoch]), train_eval, valid_eval], axis=0))
        write_logfile(epoch, record_data, f"logfile.csv")
        is_bestmodel, stopping = stop.check(epoch, valid_eval[-1])

        if is_bestmodel == True:
            bestmodel = model.state_dict().copy()
            th.save(bestmodel, f"saved_model.pth")
            best_valid_results = valid_eval.copy()
            
        if stopping == True:
            print("Earlystopping !")
            break
    
    out_results(best_valid_results, f"results.csv")