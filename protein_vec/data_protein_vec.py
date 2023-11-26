#####Data_embed_structure
from pathlib import Path
from dataclasses import dataclass
from typing import Union, List, Tuple, Any, Dict, Optional
import pickle
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict
import re
from torch import nn



class get_parquet(Dataset):
    """                                                                                                                                                                                    
    Dataset wrapper for HDF5 files                                                                                                                                                         
    """
    def __init__(self,
                 pair_path,
                 embedding_path,
                 indices: Optional[List[int]] = None
                 ):
        """                                                                                                                                                                                
        Construct the dataset                                                                                                                                                              
        :args:                                                                                                                                                                             
            :filepath - where to read from                                                                                                                                                 
        """
        self.pairs = pd.read_parquet(pair_path)
        self.embedding_path = embedding_path
        

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        sample = self.pairs.iloc[index,:]

        id_var = sample[0]
        id_path = str(self.embedding_path) + "/" + id_var[0:2] + "/" + id_var
        id_embedding = pickle.load(open(id_path, "rb"))

        total_sample = defaultdict(dict)
        total_sample['id'] = id_embedding
        total_sample['key'] = sample[2]
        total_sample['margin'] = sample[3]
        total_sample['tm'] = sample[4]
        total_sample['tm_score'] = sample[5]

        positive = sample[1].split(":")[0]
        negative = sample[1].split(":")[1]
        positive_path = str(self.embedding_path) + "/" + positive[0:2] + "/" + positive
        negative_path = str(self.embedding_path) + "/" + negative[0:2] + "/" + negative
        total_sample["positive"] = pickle.load(open(positive_path, "rb"))
        total_sample["negative"] = pickle.load(open(negative_path, "rb"))

        total_sample = dict(total_sample)
        return total_sample


def collate_fn(batch, pad_id = 0):
    all_data_dict = defaultdict(dict)
    batch_size = len(batch)
    
    keys = [t['key'] for t in batch]
    all_data_dict['key'] = keys
    
    tms = [t['tm'] for t in batch]
    all_data_dict['tm'] = tms
    
    tm_scores = [t['tm_score'] for t in batch]
    all_data_dict['tm_scores'] = torch.FloatTensor(tm_scores)
    
    margins = [t['margin'] for t in batch]
    all_data_dict['margin'] = torch.FloatTensor(margins)
    
    shape_ids = [t['id'][0].shape[0] for t in batch]
    dim = batch[0]['id'].shape[2]
    pos_shapes = [t["positive"][0].shape[0] for t in batch]
    neg_shapes = [t["negative"][0].shape[0] for t in batch]
    shape_ids += pos_shapes
    shape_ids += neg_shapes

    biggest_shape = np.max(shape_ids)
    pad_tensor = torch.zeros(biggest_shape, dim).type(torch.BoolTensor)


    id_tensors = [t['id'][0] for t in batch]
    id_tensors.append(pad_tensor)
    padded_ids = torch.nn.utils.rnn.pad_sequence(id_tensors, padding_value=0, batch_first=True)[:-1, :, :]
    id_padding = torch.zeros(padded_ids.shape[0:2]).type(torch.BoolTensor)
    id_padding[padded_ids[:,:,0] == pad_id] = True
    all_data_dict['id'] = padded_ids
    all_data_dict['id_padding'] = id_padding

    pos_tensor_values = [t["positive"][0] for t in batch]
    pos_tensor_values.append(pad_tensor)
    neg_tensor_values = [t["negative"][0] for t in batch]
    neg_tensor_values.append(pad_tensor)

    padded_pos_tensors = torch.nn.utils.rnn.pad_sequence(pos_tensor_values, padding_value=0, batch_first=True)[:-1, :, :]
    padded_neg_tensors = torch.nn.utils.rnn.pad_sequence(neg_tensor_values, padding_value=0, batch_first=True)[:-1, :, :]

    pos_pad_labels = torch.zeros(padded_pos_tensors.shape[0:2]).type(torch.BoolTensor)
    pos_pad_labels[padded_pos_tensors[:,:,0] == pad_id] = True
    neg_pad_labels = torch.zeros(padded_neg_tensors.shape[0:2]).type(torch.BoolTensor)
    neg_pad_labels[padded_neg_tensors[:,:,0] == pad_id] = True
    
    all_data_dict["positive"] = padded_pos_tensors
    all_data_dict["negative"] = padded_neg_tensors
    all_data_dict["positive_padding"] = pos_pad_labels
    all_data_dict["negative_padding"] = neg_pad_labels
    
    all_data_dict = dict(all_data_dict)

    return(all_data_dict)




#Construct datasets function
def construct_datasets(pair_path, embedding_path, train_prop=.9, val_prop=.05, test_prop = .05):
    dataset = get_parquet(pair_path, embedding_path)
    total_samples = len(dataset)
    sampleable_values = np.arange(total_samples)

    train_n_to_sample = int(len(sampleable_values) * train_prop)
    val_n_to_sample = int(len(sampleable_values) * val_prop)
    test_n_to_sample = int(len(sampleable_values) * test_prop)

    train_indices = np.random.choice(sampleable_values, train_n_to_sample, replace=False)
    sampleable_values = sampleable_values[~np.isin(sampleable_values, train_indices)]
    val_indices = np.random.choice(sampleable_values, val_n_to_sample, replace=False)
    sampleable_values = sampleable_values[~np.isin(sampleable_values, val_indices)]
    test_indices = np.random.choice(sampleable_values, test_n_to_sample, replace=False)

    #Make train, test, and validation datasets using torch subset
    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds =  torch.utils.data.Subset(dataset, val_indices)
    test_ds = torch.utils.data.Subset(dataset, test_indices)
    
    return(train_ds, val_ds, test_ds)
