import pandas as pd
import torch as th
import os, sys
from Bio import SeqIO
import numpy as np
import gc, re

max_length = 1000

def get_lm_embedding(inp_seq, tokenizer, model, device):
    model = model.eval()
    if len(inp_seq) > max_length:
        src_list = [x for x in range(0, len(inp_seq), int(max_length/2))][:-1]
        embed_dict = {}
        keys = []
        for num, src in enumerate(src_list):
            key = f"{src}-{src+int(max_length/2)}"
            if key not in keys:
                keys.append(key)
                
            if num == len(src_list) - 1:
                seq = inp_seq[src:]
                key = f"{src+int(max_length/2)}-"
                if key not in keys:
                    keys.append(key)
            else:
                key = f"{src+int(max_length/2)}-{src+max_length}"
                if key not in keys:
                    keys.append(key)
                
                seq = inp_seq[src:src+max_length]
            seq_str = " ".join(seq)

            new_seq = [re.sub(r"[UZOB]", "X", seq_str)]
            ids = tokenizer.batch_encode_plus(new_seq, add_special_tokens=True, padding=True)
            input_ids = th.tensor(ids['input_ids']).to(device)
            attention_mask = th.tensor(ids['attention_mask']).to(device)
            with th.no_grad():
                embedding = model(input_ids=input_ids, attention_mask=attention_mask)
            last_embed = embedding.last_hidden_state.cpu().numpy()[:, :-1, :][0] # [L, 1024]

            if num != len(src_list) - 1:
                key = f"{src}-{src+int(max_length/2)}"
                if not key in list(embed_dict.keys()):
                    embed_dict[key] = [last_embed[:int(max_length/2)]]
                else:
                    embed_dict[key].append(last_embed[:int(max_length/2)])
                
                key = f"{src+int(max_length/2)}-{src+max_length}"
                if not key in list(embed_dict.keys()):
                    embed_dict[key] = [last_embed[int(max_length/2):]]
                else:
                    embed_dict[key].append(last_embed[int(max_length/2):])

            else:
                key = f"{src}-{src+int(max_length/2)}"
                if not key in list(embed_dict.keys()):
                    embed_dict[key] = [last_embed[:int(max_length/2)]]
                else:
                    embed_dict[key].append(last_embed[:int(max_length/2)])

                key = f"{src+int(max_length/2)}-"
                embed_dict[key] = [last_embed[int(max_length/2):]]

        embed_list = []
        for key in keys:
            embed = np.array(embed_dict[key]).mean(0)
            embed_list.append(embed)
        embed_array = np.concatenate(embed_list, axis=0)
    
    else:
        seq_str = " ".join(inp_seq)
        new_seq = [re.sub(r"[UZOB]", "X", seq_str)]
        ids = tokenizer.batch_encode_plus(new_seq, add_special_tokens=True, padding=True)
        input_ids = th.tensor(ids['input_ids']).to(device)
        attention_mask = th.tensor(ids['attention_mask']).to(device)
        with th.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embed_array = embedding.last_hidden_state.cpu().numpy()[:, :-1, :][0] # [L, 1024]

    return embed_array
