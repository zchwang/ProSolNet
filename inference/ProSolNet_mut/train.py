import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, Dataset
from utils import EarlyStopping, run_a_training_epoch, run_an_eval_epoch, \
        write_logfile, evaluate, recover_pred
from argparse import RawDescriptionHelpFormatter
import argparse
import os
import datetime
from model import SolubilityNet
import copy

class Datasets(Dataset):
    def __init__(self, keys, seq_feat, feat_list, xyz_list, surf_feat_list, surf_res, lm_list,
                 wt_seq_feat, wt_feat_list, wt_xyz_list, wt_surf_feat_list, wt_surf_res, wt_lm_list, label):
        self.keys = keys
        self.seq_feat = seq_feat
        self.feat_list = feat_list
        self.xyz_list = xyz_list
        self.surf_feat_list = surf_feat_list
        self.surf_res = surf_res
        self.lm_list = lm_list
        self.wt_seq_feat = wt_seq_feat
        self.wt_feat_list = wt_feat_list
        self.wt_xyz_list = wt_xyz_list
        self.wt_surf_feat_list = wt_surf_feat_list
        self.wt_surf_res = wt_surf_res
        self.wt_lm_list = wt_lm_list
        self.label = label

    def __getitem__(self, idx):
        return self.keys[idx], self.seq_feat[idx].unsqueeze(0), self.feat_list[idx], self.xyz_list[idx], \
            self.surf_feat_list[idx], self.surf_res[idx], self.lm_list[idx], self.wt_seq_feat[idx].unsqueeze(0), self.wt_feat_list[idx], \
            self.wt_xyz_list[idx], self.wt_surf_feat_list[idx], self.wt_surf_res[idx], self.wt_lm_list[idx], self.label[idx].unsqueeze(0)

    def __len__(self):
        return len(self.keys)
    
    def _get_data(self, lm_feats, seq_feats, feats, xyz, surf_feat, surf_res):
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
        seq_feats = th.cat(seq_feats, axis=0)
        feats = th.cat(feats, axis=0)
        lm_feats = th.cat(lm_feats, axis=0)
        xyz_tensor = th.cat(xyz, axis=0)

        return lm_feats, seq_feats, feats, xyz_tensor, final_surf_feat, final_surf_res, res_batch, batch

    def get_loss_weight(self, keys, labels, _weight=1.5):
        weights = []
        for k, l in zip(keys, labels):
            if k.endswith("reverse"):
                if l == -1:
                    weights.append(_weight)
                else:
                    weights.append(1)
            else:
                if l == 1:
                    weights.append(_weight)
                else:
                    weights.append(1)
        weights = th.tensor(weights).to(th.float32)
        return weights

    def data_collate(self, data):
        keys, seq_feats, feats, xyz, surf_feat, surf_res, lm_feats, wt_seq_feats, wt_feats, wt_xyz, wt_surf_feat, wt_surf_res, wt_lm_feats, labels = zip(*data)

        lm_feats, seq_feats, feats, xyz_tensor, final_surf_feat, final_surf_res, res_batch, batch = \
            self._get_data(lm_feats, seq_feats, feats, xyz, surf_feat, surf_res)
        wt_lm_feats, wt_seq_feats, wt_feats, wt_xyz_tensor, wt_final_surf_feat, wt_final_surf_res, wt_res_batch, wt_batch = \
            self._get_data(wt_lm_feats, wt_seq_feats, wt_feats, wt_xyz, wt_surf_feat, wt_surf_res)

        labels = th.cat(labels, axis=0)
        weights = self.get_loss_weight(keys, labels)

        return {"keys": keys, "weights": weights, "seq_feats": seq_feats, "feats": feats, "xyz": xyz_tensor, "surf_feats": final_surf_feat,
                "surf_res": final_surf_res, "lm_feat": lm_feats, "res_batch": res_batch, "batch": batch, "wt_seq_feats": wt_seq_feats, "wt_feats": wt_feats,
                "wt_xyz": wt_xyz_tensor, "wt_surf_feats": wt_final_surf_feat, "wt_surf_res": wt_final_surf_res, "wt_lm_feat": wt_lm_feats, "wt_res_batch": wt_res_batch,
                "wt_batch": wt_batch, "labels": labels}

class ProteinDataset():
    def __init__(self, seq_fpath, graph_fpath, surf_fpath, lm_fpath, split_fpath, label_fpath, test_inp, batch_size):
        self.batch_size = batch_size
        seq_df = pd.read_pickle(seq_fpath)
        self.seq_dict = dict(zip(seq_df.index.tolist(), seq_df.values))
        self.graph_dict = np.load(graph_fpath, allow_pickle=True).item()
        self.surf_dict = np.load(surf_fpath, allow_pickle=True).item()
        self.lm_dict = np.load(lm_fpath, allow_pickle=True).item()
        self.label_dict = np.load(label_fpath, allow_pickle=True).item()
        self.data_split_dict = np.load(split_fpath, allow_pickle=True).item()
        with open(test_inp) as f:
            self.test_keys = [x.strip() for x in f.readlines()]

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

    def process_feats(self, keys, reverse, shuffle):
        seq_feat_list = []
        res_feat_list = []
        surf_feat_list = []
        surf_res_index = []
        lm_feat_list = []
        xyz_list = []
        label_list = []

        wt_seq_feat_list = []
        wt_res_feat_list = []
        wt_surf_feat_list = []
        wt_surf_res_index = []
        wt_lm_feat_list = []
        wt_xyz_list = []

        final_keys = []
        for key in keys:
            if key in list(self.graph_dict.keys()) and key in list(self.surf_dict.keys()):
                if key.startswith("325534072"):
                    continue
                d = self.graph_dict[key]
                resid, hb, xyz, dssp, sas = list(d.values())
                if len(xyz) > 1000:
                    continue
                final_keys.append(key)
                surf_d = self.surf_dict[key]
                surf_feat_list.append(th.from_numpy(surf_d["feat"]).to(th.float32))
                surf_res_index.append(th.from_numpy(surf_d["res_idx"]).to(th.float32))
                lm_feat_list.append(th.from_numpy(self.lm_dict[key]).to(th.float32))
                seq_feat = th.from_numpy(self.seq_dict[key])
                seq_feat_list.append(seq_feat.reshape(1, -1))
                
                res_one_hot = th.cat([self.get_one_hot(res, self.aa_list).reshape(1, -1) for res in resid], axis=0).to(th.float32)
                dssp_one_hot = th.cat([self.get_one_hot(d, self.dssp_list).reshape(1, -1) for d in dssp], axis=0).to(th.float32)
                phy_chem_feat = th.cat([self.get_phy_chem_feat(i).reshape(1, -1) for i in resid], axis=0)
                res_feat_list.append(th.cat([res_one_hot, phy_chem_feat, dssp_one_hot], axis=-1))
                xyz_list.append(th.from_numpy(xyz).to(th.float32))
                label_list.append(self.label_dict[key])

                wt_k = key.split("_")[0] + "-wt"
                wt_d = self.graph_dict[wt_k]
                wt_surf_d = self.surf_dict[wt_k]
                wt_surf_feat_list.append(th.from_numpy(wt_surf_d["feat"]).to(th.float32))
                wt_surf_res_index.append(th.from_numpy(wt_surf_d["res_idx"]).to(th.float32))
                wt_lm_feat_list.append(th.from_numpy(self.lm_dict[wt_k]).to(th.float32))
                wt_seq_feat = th.from_numpy(self.seq_dict[wt_k])
                wt_seq_feat_list.append(wt_seq_feat.reshape(1, -1))
                resid, hb, wt_xyz, dssp, sas = list(wt_d.values())
                wt_res_one_hot = th.cat([self.get_one_hot(res, self.aa_list).reshape(1, -1) for res in resid], axis=0).to(th.float32)
                wt_dssp_one_hot = th.cat([self.get_one_hot(_d, self.dssp_list).reshape(1, -1) for _d in dssp], axis=0).to(th.float32)
                wt_phy_chem_feat = th.cat([self.get_phy_chem_feat(i).reshape(1, -1) for i in resid], axis=0)
                wt_res_feat_list.append(th.cat([wt_res_one_hot, wt_phy_chem_feat, wt_dssp_one_hot], axis=-1))
                wt_xyz_list.append(th.from_numpy(wt_xyz).to(th.float32))

                if reverse:
                    seq_feat_list.append(wt_seq_feat.reshape(1, -1))
                    res_feat_list.append(th.cat([wt_res_one_hot, wt_phy_chem_feat, wt_dssp_one_hot], axis=-1))
                    surf_feat_list.append(th.from_numpy(wt_surf_d["feat"]).to(th.float32))
                    surf_res_index.append(th.from_numpy(wt_surf_d["res_idx"]).to(th.float32))
                    lm_feat_list.append(th.from_numpy(self.lm_dict[wt_k]).to(th.float32))
                    xyz_list.append(th.from_numpy(wt_xyz).to(th.float32))
                    wt_seq_feat_list.append(seq_feat.reshape(1, -1))
                    wt_res_feat_list.append(th.cat([res_one_hot, phy_chem_feat, dssp_one_hot], axis=-1))
                    wt_surf_feat_list.append(th.from_numpy(surf_d["feat"]).to(th.float32))
                    wt_surf_res_index.append(th.from_numpy(surf_d["res_idx"]).to(th.float32))
                    wt_lm_feat_list.append(th.from_numpy(self.lm_dict[key]).to(th.float32))
                    wt_xyz_list.append(th.from_numpy(xyz).to(th.float32))

                    label_list.append(self.label_dict[key] * -1)
                    final_keys.append(key + "_reverse")

        seq_feat = th.cat(seq_feat_list, axis=0).to(th.float32)
        wt_seq_feat = th.cat(wt_seq_feat_list, axis=0).to(th.float32)
       
        label_tensor = th.from_numpy(np.array(label_list)).to(th.float32)
        dataset = Datasets(final_keys, seq_feat, res_feat_list, xyz_list, surf_feat_list, surf_res_index, lm_feat_list,
                           wt_seq_feat, wt_res_feat_list, wt_xyz_list, wt_surf_feat_list, wt_surf_res_index, wt_lm_feat_list, label_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=dataset.data_collate, num_workers=2)

        return dataloader, label_tensor.numpy()
    
    def get_test_dataloader(self):
        test_dataloader, test_label = self.process_feats(self.test_keys, False, False)
        return test_dataloader, test_label

    def get_dataloader(self, idx):
        train_keys = self.data_split_dict[idx]["train"]
        valid_keys = self.data_split_dict[idx]["valid"]
        print("train number:", len(train_keys))
        print("valid number:", len(valid_keys))
        train_dataloader, _ = self.process_feats(train_keys, True, True)
        valid_dataloader, _ = self.process_feats(valid_keys, False, False)

        return train_dataloader, valid_dataloader

if __name__ == "__main__":
    d = "Training ..."
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-graph_fpath", type=str, default="train_esm.pkl",
                         help="Input. The path of training data.")
    parser.add_argument("-seq_fpath", type=str, default="train_esm.pkl",
                         help="Input. The path of training data.")
    parser.add_argument("-surf_fpath", type=str, default="train_esm.pkl",
                         help="Input. The path of training data.")
    parser.add_argument("-label_fpath", type=str, default="train_esm.pkl",
                         help="Input. The path of training data.")
    parser.add_argument("-lm_fpath", type=str, default="train_esm.pkl",
                         help="Input. The path of training data.")
    parser.add_argument("-split_fpath", type=str, default="train_esm.pkl",
                         help="Input. The path of training data.")
    parser.add_argument("-test_inp", type=str, default="train_esm.pkl",
                         help="Input. The path of training data.")
    parser.add_argument("-batch_size", type=int, default=64,
                        help="Input. Batch size")
    parser.add_argument("-hid_dim", type=int, default=64,
                        help="Input. Batch size")
    parser.add_argument("-lr", type=float, default=0.001,
                        help="Input. Learning rate.")
    parser.add_argument("-drop_rate", type=float, default=0.0,
                        help="Input. The rate of dropout.")
    parser.add_argument("-epochs", type=int, default=150,
                        help="Input. Epochs")
    parser.add_argument("-device", type=str, default="cuda",
                        help="Input. The device: cuda or cpu.")
    args = parser.parse_args()
    graph_fpath = args.graph_fpath
    seq_fpath = args.seq_fpath
    surf_fpath = args.surf_fpath
    lm_fpath = args.lm_fpath
    label_fpath = args.label_fpath
    split_fpath = args.split_fpath
    test_inp = args.test_inp
    lr = args.lr 
    batch_size = args.batch_size 
    hid_dim = args.hid_dim
    drop_rate = args.drop_rate
    device = args.device 
    epochs = args.epochs
    patience = 10
    min_delta = 0.001
    data = ProteinDataset(seq_fpath, graph_fpath, surf_fpath, lm_fpath, split_fpath, label_fpath, test_inp, batch_size)
    test_dataloader, test_label_array = data.get_test_dataloader()

    all_valid_pred_labels = []
    all_valid_keys = []
    all_test_pred = []
    for fold in range(1, 11):
        if os.path.exists(f"eval-{fold}.csv"):
            print(f"eval-{fold}.csv existed ...")
            continue
        print(f"fold: {fold}/10")
        train_dataloader, valid_dataloader = data.get_dataloader(fold)
        model = SolubilityNet(hid_dim=hid_dim, drop_rate=drop_rate).to(device)
        optimizer = th.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
        stop = EarlyStopping(patience, min_delta)
        bestmodel = None
        record_data = []
        start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        best_results = None
        valid_pred_label_array = None
        test_pred_label_array = None
        for epoch in range(epochs):
            print("Epoch: {}".format(epoch))
            train_eval = run_a_training_epoch(model, train_dataloader, optimizer, device)
            valid_keys, valid_pred, valid_real, valid_eval = run_an_eval_epoch(model, valid_dataloader, device)
            test_keys, test_pred, test_real, test_eval = run_an_eval_epoch(model, test_dataloader, device)
            record_data.append(np.concatenate([np.array([epoch]), train_eval, valid_eval, test_eval], axis=0))
            write_logfile(epoch, record_data, f"logfile_{fold}")
            is_bestmodel, stopping = stop.check(epoch, -valid_eval[1])

            if is_bestmodel == True:
                bestmodel = model.state_dict().copy()
                th.save(bestmodel, f"model-{fold}.pth")
                best_results = np.concatenate([valid_eval.reshape(1, -1), test_eval.reshape(1, -1)], axis=0)
                valid_pred_label_array = np.concatenate([valid_pred.reshape(-1, 1), valid_real.reshape(-1, 1)], axis=1)
                test_pred_label_array = np.concatenate([test_pred.reshape(-1, 1), test_real.reshape(-1, 1)], axis=1)
                test_pred_array = test_pred.copy()
         

            if stopping == True or epoch == (epochs-1):
                all_valid_pred_labels.append(valid_pred_label_array)
                all_valid_keys += valid_keys
                all_test_pred.append(test_pred_array.reshape(-1, 1))
                eval_df = pd.DataFrame(best_results, index=["valid", "test"], columns=["ACC", "norm-ACC", "macro-f1", "loss"])
                eval_df.to_csv(f"eval-{fold}.csv")
                test_df = pd.DataFrame(test_pred_label_array, index=test_keys, columns=["pred", "label"])
                test_df.to_csv(f"{fold}-test_pred-label.csv")
                print("Earlystopping !")
                break
        
        end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(f'time-{fold}.dat', 'w') as f:
            f.writelines('Start Time:  ' + start_time + '\n')
            f.writelines('End Time:  ' + end_time)

    all_valid_pred_labels_array = np.concatenate(all_valid_pred_labels, axis=0)
    valid_pred_label_df = pd.DataFrame(all_valid_pred_labels_array, index=all_valid_keys, columns=["pred", "label"])
    valid_pred_label_df.to_csv("valid_pred-label.csv")

    test_mean_pred = np.mean(np.concatenate(all_test_pred, axis=1), axis=1)
    test_mean_pred_label = np.concatenate([test_mean_pred.reshape(-1, 1), test_label_array.reshape(-1, 1)], axis=1)
    test_mean_pred_label_df = pd.DataFrame(test_mean_pred_label, index=test_keys, columns=["pred", "label"])
    test_mean_pred_label_df.to_csv("final_test-pred-label.csv")

    valid_eval = evaluate(all_valid_pred_labels_array[:, 1], recover_pred(all_valid_pred_labels_array[:, 0]))
    test_eval = evaluate(test_label_array.ravel(), recover_pred(test_mean_pred.ravel()))
    valid_test_eval = np.concatenate([valid_eval.reshape(1, -1), test_eval.reshape(1, -1)], axis=0)
    final_eval_df = pd.DataFrame(valid_test_eval, index=["valid", "mean-test"], columns=["ACC", "norm-ACC", "macro-f1"])
    final_eval_df.to_csv("results.csv")
