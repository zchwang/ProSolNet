import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
import os

def evaluate(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred_binary)
    tp = np.sum(y_true * y_pred_binary, axis=0)
    tn = np.sum((1 - y_true) * (1 - y_pred_binary), axis=0)
    fp = np.sum((1 - y_true) * y_pred_binary, axis=0)
    fn = np.sum(y_true * (1 - y_pred_binary), axis=0)
    sensitivity = np.round(tp/(tp+fn), 2)
    specificity = np.round(tn/(tn+fp), 2)

    return np.round(np.array([auc, accuracy, sensitivity, specificity]), 3)

def out_results(values, file_):
    columns = ["AUC", "ACC", "SN", "SP", "loss"]
    df = pd.DataFrame(values.reshape(1, -1), index=["valid"], columns=columns)
    df.to_csv(file_, float_format="%.5f")

def write_logfile(epoch, record_data, logfile):
    if epoch == 0:
        if os.path.exists(logfile):
            os.remove(logfile)

    index = [x for x in range(epoch + 1)]
    values = np.array(record_data).reshape(epoch+1, -1)
    columns = ["epoch", "T_AUC", "T_ACC", "T_SN", "T_SP", "T_loss", 
                        "V_AUC", "V_ACC", "V_SN", "V_SP", "V_loss"]
    df = pd.DataFrame(values, index=index, columns=columns)
    df.to_csv(logfile, float_format="%.4f")


BCELoss = th.nn.BCELoss()
def run_a_training_epoch(model, dataloader, optimizer, device):
    model.train()
    pred_list = []
    real_list = []
    total_loss = 0
    for step, data in enumerate(dataloader):
        keys, seq_feats, feats, xyz, surf_feats, surf_res, labels, res_batch, batch = list(data.values())
        seq_feats = seq_feats.to(device)
        feats = feats.to(device)
        xyz = xyz.to(device)
        surf_feats = surf_feats.to(device)
        surf_res = surf_res.to(device)
        labels = labels.to(device)
        res_batch = res_batch.to(device)
        batch = batch.to(device)
        if len(labels) == 1:
            continue
        _, pred_proba = model(seq_feats, feats, xyz, surf_feats, surf_res, res_batch, batch)
        loss = BCELoss(pred_proba.ravel(), labels.ravel())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.cpu().detach().numpy()
        pred_list.append(pred_proba.cpu().detach().numpy().ravel())
        real_list.append(labels.cpu().detach().numpy().ravel())

    N = len(dataloader)
    pred_array = np.concatenate(pred_list, axis=0)
    real_array = np.concatenate(real_list, axis=0)
    eval = evaluate(real_array, pred_array)

    return np.concatenate([eval, np.array([np.round(total_loss/N, 4)])], axis=0)

def run_an_eval_epoch(model, dataloader, device):
    model.eval()
    with th.no_grad():
        total_loss = 0
        pred_list = []
        real_list = []
        for step, data in enumerate(dataloader):
            keys, seq_feats, feats, xyz, surf_feats, surf_res, labels, res_batch, batch = list(data.values())
            seq_feats = seq_feats.to(device)
            feats = feats.to(device)
            xyz = xyz.to(device)
            surf_feats = surf_feats.to(device)
            surf_res = surf_res.to(device)
            labels = labels.to(device)
            res_batch = res_batch.to(device)
            batch = batch.to(device)
            _, pred_proba = model(seq_feats, feats, xyz, surf_feats, surf_res, res_batch, batch)
            loss = BCELoss(pred_proba.ravel(), labels.ravel())

            total_loss += loss.cpu().detach().numpy()
            pred_list.append(pred_proba.cpu().detach().numpy().ravel())
            real_list.append(labels.cpu().detach().numpy().ravel())
        
        pred_array = np.concatenate(pred_list, axis=0)
        real_array = np.concatenate(real_list, axis=0)
        eval = evaluate(real_array, pred_array)
        N = len(dataloader)

        return pred_array, real_array, np.concatenate([eval, np.array([np.round(total_loss/N, 4)])], axis=0)

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
