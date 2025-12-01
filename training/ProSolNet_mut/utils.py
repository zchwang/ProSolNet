import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import os

def evaluate(y_true, pred_idx):
    accuracy = accuracy_score(y_true, pred_idx)
    normal_accuracy = balanced_accuracy_score(y_true, pred_idx)
    macro_f1 = f1_score(y_true, pred_idx, average="macro")
    return np.array([accuracy, normal_accuracy, macro_f1])

def write_logfile(epoch, record_data, logfile):
    if epoch == 0:
        if os.path.exists(logfile):
            os.remove(logfile)

    index = [x for x in range(epoch + 1)]
    values = np.array(record_data).reshape(epoch+1, -1)
    columns = ["epoch", "T_ACC", "T_norm-ACC", "T_macro-f1", "T_loss", "V_ACC", "V_norm-ACC", "V_macro-f1", "V_loss"]
    df = pd.DataFrame(values, index=index, columns=columns)
    df.to_csv(logfile, float_format="%.4f")

def recover_pred(pred_tensor):
    interval = 2/3
    pred_1 = (pred_tensor >= -1.) * (pred_tensor < interval-1) * -1
    pred_2 = (pred_tensor >= interval-1) * (pred_tensor < 2*interval-1) * 0
    pred_3 = (pred_tensor >= 2*interval-1) * (pred_tensor <= 1) * 1
    return (pred_1 + pred_2 + pred_3)
    
BCELoss = th.nn.BCELoss()
def run_a_training_epoch(model, dataloader, optimizer, device):
    model.train()
    pred_list = []
    real_list = []
    total_loss = 0
    for step, data in enumerate(dataloader):
        keys, weights, feats, xyz, surf_feats, surf_res, lm_feats, res_batch, batch, wt_feats, \
            wt_xyz, wt_surf_feats, wt_surf_res, wt_lm_feats, wt_res_batch, wt_batch, labels = list(data.values())
        weights = weights.to(device)
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
        label = labels.to(device)
        if len(labels) == 1:
            continue
        _, pred_proba = model(feats, xyz, surf_feats, surf_res, lm_feats, res_batch, batch,
                            wt_feats, wt_xyz, wt_surf_feats, wt_surf_res, wt_lm_feats, wt_res_batch, wt_batch)
      
        loss = th.mean(weights * th.square(pred_proba.ravel() - label.ravel())) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.cpu().detach().numpy()
        pred_list.append(pred_proba.cpu().detach())
        real_list.append(label.cpu().detach().numpy().ravel())

    N = len(dataloader)
    pred_tensor = th.cat(pred_list, axis=0)
    pred_idx = recover_pred(pred_tensor.cpu().detach().numpy())
    real_array = np.concatenate(real_list, axis=0)
    eval = evaluate(real_array, pred_idx)

    return np.concatenate([eval, np.array([np.round(total_loss/N, 4)])], axis=0)

def run_an_eval_epoch(model, dataloader, device):
    model.eval()
    with th.no_grad():
        all_keys = []
        total_loss = 0
        pred_list = []
        real_list = []
        for step, data in enumerate(dataloader):
            keys, weights, feats, xyz, surf_feats, surf_res, lm_feats, res_batch, batch, wt_feats, \
                wt_xyz, wt_surf_feats, wt_surf_res, wt_lm_feats, wt_res_batch, wt_batch, labels = list(data.values())
            all_keys += keys
            weights = weights.to(device)
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
            label = labels.to(device)
            _, pred_proba = model(feats, xyz, surf_feats, surf_res, lm_feats, res_batch, batch,
                            wt_feats, wt_xyz, wt_surf_feats, wt_surf_res, wt_lm_feats, wt_res_batch, wt_batch)
  
            loss = th.mean(weights * th.square(pred_proba.ravel() - label.ravel()))
            total_loss += loss.cpu().detach().numpy()
            pred_list.append(pred_proba.cpu().detach().numpy().ravel())
            real_list.append(label.cpu().detach().numpy().ravel())
        
        N = len(dataloader)
        pred_array = np.concatenate(pred_list, axis=0)
        pred_idx = recover_pred(pred_array)
        real_array = np.concatenate(real_list, axis=0)
        eval = evaluate(real_array, pred_idx)

        return all_keys, pred_array, real_array, np.concatenate([eval, np.array([np.round(total_loss/N, 4)])], axis=0)

class EarlyStopping(object):
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.min_loss = 99.
        self.count_epoch = 0
        self.stop = False
        self.is_bestmodel = False

    def check(self, epoch, cur_loss):
        if epoch == 0:
            self.min_loss = cur_loss
            self.count_epoch += 1
            self.is_bestmodel = True
        else:
            if cur_loss < self.min_loss - self.min_delta:
                self.min_loss = cur_loss
                self.count_epoch = 0
                self.is_bestmodel = True
            else:
                self.count_epoch += 1
                self.is_bestmodel = False

        if self.count_epoch == self.patience:
            self.stop = True

        return self.is_bestmodel, self.stop
