import torch as th
from torch import nn
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from torch_geometric.utils import softmax
import torch.nn.functional as F

class update_v(th.nn.Module):
    def __init__(self, hid_dim):
        super(update_v, self).__init__()
        self.a = nn.Linear(hid_dim*2, 1, bias=False)
        self.lin_layer = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim), nn.LeakyReLU())
        for layer in self.lin_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
    
    def forward(self, feat, edge_index):
        u, v = edge_index
        u_feat = feat[u]
        v_feat = feat[v]
        N, dim = feat.shape
        m = th.cat([u_feat, v_feat], axis=1)
        attn = self.a(m)
        weight = softmax(attn, v, num_nodes=N)
        out = th.zeros(N, dim).to(feat.device)
        new_feat = scatter(u_feat*weight, v, dim=0, out=out, reduce="sum")
        new_feat = self.lin_layer(new_feat)

        return new_feat

class ProteinGraph(nn.Module):
    def __init__(self, cutoff=5.0, num_layers=4, hid_dim=64):
        super(ProteinGraph, self).__init__()
        self.cutoff = cutoff
        self.encoder = nn.Sequential(nn.Linear(17+21, hid_dim), nn.BatchNorm1d(hid_dim))
        self.update_vs = nn.ModuleList([update_v(hid_dim) for _ in range(num_layers)])
        self.prot_bn_act = nn.Sequential(nn.BatchNorm1d(hid_dim), nn.LeakyReLU())
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for layers in [self.encoder]:
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    layer.bias.data.fill_(0)
             
    def forward(self, edge_index, n_feats, batch):
        n_feats = self.encoder(n_feats)
        feat_list = [n_feats.unsqueeze(1)]
        for layer in self.update_vs:
            n_feats = layer(n_feats, edge_index)
            feat_list.append(n_feats.unsqueeze(1))
        feats = th.cat(feat_list, axis=1)
        n_feats = th.mean(feats, axis=1)
        prot_feats = scatter(n_feats, batch, dim=0, reduce="mean")
        prot_feats_2 = self.prot_bn_act(prot_feats)

        return prot_feats_2

class ProteinLMGraph(nn.Module):
    def __init__(self, cutoff=5.0, num_layers=4, hid_dim=64):
        super(ProteinLMGraph, self).__init__()
        self.cutoff = cutoff
        self.surf_up_lin = nn.Sequential(nn.Linear(3, hid_dim), nn.BatchNorm1d(hid_dim), nn.LeakyReLU())

        self.update_vs = nn.ModuleList([update_v(1024) for _ in range(num_layers)])
        self.hid_lin = nn.Sequential(nn.Linear(1024, hid_dim))
        self.surf_down_lin = nn.Sequential(nn.Linear(hid_dim*2, hid_dim), nn.BatchNorm1d(hid_dim), nn.LeakyReLU())
    
        self.prot_bn_act = nn.Sequential(nn.LayerNorm(hid_dim), nn.LeakyReLU())
        self.surf_bn_act = nn.Sequential(nn.BatchNorm1d(hid_dim), nn.LeakyReLU())

        self.reset_parameters()
    
    def reset_parameters(self):
        for layers in [self.hid_lin, self.surf_up_lin, self.surf_down_lin]:
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    layer.bias.data.fill_(0)
    
    def forward(self, edge_index, n_feats, batch, surf_feats, surf_res, res_batch):
        surf_feats = self.surf_up_lin(surf_feats)

        surf_res_set = th.unique(surf_res)
        surf_dim = surf_feats.shape[-1]
        out = th.zeros(len(n_feats), surf_dim).to(n_feats.device)
        surf_feats = scatter(surf_feats, surf_res, dim=0, out=out, reduce="mean")
        surf_feats = surf_feats[surf_res_set]

        feat_list = [n_feats.unsqueeze(1)]
        for layer in self.update_vs:
            n_feats = layer(n_feats, edge_index)
            feat_list.append(n_feats.unsqueeze(1))
        feats = th.cat(feat_list, axis=1)
        n_feats = th.mean(feats, axis=1)
        n_feats = self.hid_lin(n_feats)
        prot_feats = scatter(n_feats, batch, dim=0, reduce="mean")
        prot_feats_2 = self.prot_bn_act(prot_feats)

        n_surf_feats = n_feats[surf_res_set]
        n_surf_feats = th.cat([n_surf_feats, surf_feats], axis=-1)
        n_surf_feats = self.surf_down_lin(n_surf_feats)
        n_surf_feats = scatter(n_surf_feats, res_batch, dim=0, reduce="mean")
        n_surf_feats_2 = self.surf_bn_act(n_surf_feats)

        return prot_feats_2, n_surf_feats_2

class SolubilityNet(nn.Module):
    def __init__(self, cutoff=8.0, num_layers=1, hid_dim=64, drop_rate=0.0):
        super(SolubilityNet, self).__init__()
        self.cutoff = cutoff
        self.prot_model = ProteinGraph(cutoff=cutoff, num_layers=num_layers, hid_dim=hid_dim)
        self.protlm_model = ProteinLMGraph(cutoff=cutoff, num_layers=num_layers, hid_dim=hid_dim)
        self.hid_lin = nn.Sequential(nn.Linear(3*hid_dim, hid_dim), nn.Dropout(p=drop_rate),
                                     nn.Linear(hid_dim, 1), nn.Tanh())
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.hid_lin:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)

    def forward(self, n_feats, pos, surf_feats, surf_res, lm_feats, res_batch, batch,
            wt_n_feats, wt_pos, wt_surf_feats, wt_surf_res, wt_lm_feats, wt_res_batch, wt_batch):
        
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        wt_edge_index = radius_graph(wt_pos, r=self.cutoff, batch=wt_batch)

        protlm_feats, surf_feats = self.protlm_model(edge_index, lm_feats, batch, surf_feats, surf_res, res_batch)
        wt_protlm_feats, wt_surf_feats = self.protlm_model(wt_edge_index, wt_lm_feats, wt_batch, wt_surf_feats, wt_surf_res, wt_res_batch)
        
        prot_feats = self.prot_model(edge_index, n_feats, batch)
        wt_prot_feats = self.prot_model(wt_edge_index, wt_n_feats, wt_batch)
        
        mut_feats = th.cat([prot_feats, surf_feats, protlm_feats], axis=1)
        wt_feats = th.cat([wt_prot_feats, wt_surf_feats, wt_protlm_feats], axis=1)

        feats = mut_feats - wt_feats
        pred = self.hid_lin(feats)
        
        return feats, pred
