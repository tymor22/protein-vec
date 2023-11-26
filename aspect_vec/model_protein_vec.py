import json
import inspect
from functools import partial
from dataclasses import dataclass, asdict

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import random



@dataclass
class Config:
    def isolate(self, config):
        specifics = inspect.signature(config).parameters
        my_specifics = {k: v for k, v in asdict(self).items() if k in specifics}
        return config(**my_specifics)

    def to_json(self, filename):
        config = json.dumps(asdict(self), indent=2)
        with open(filename, 'w') as f:
            f.write(config)
    
    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as f:
            js = json.loads(f.read())
        config = cls(**js)
        return config
    

@dataclass
class trans_basic_block_Config(Config):
    d_model: int = 1024
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 2048
    out_dim: int = 512
    dropout: float = 0.1
    activation: str = 'relu'
    num_variables: int = 10 #9
    vocab: int = 20
    # data params
    lr0: float = 0.0001
    warmup_steps: int = 300
    p_bernoulli: float = .5
    
    def build(self):
        return trans_basic_block(self)

class trans_basic_block(pl.LightningModule):
    """
    TransformerEncoderLayer with preset parameters followed by global pooling and dropout
    """
    def __init__(self, config: trans_basic_block_Config):
        super().__init__()
        self.config = config

        #Encoding
        encoder_args = {k: v for k, v in asdict(config).items() if k in inspect.signature(nn.TransformerEncoderLayer).parameters} 
        num_layers = config.num_layers
        encoder_layer = nn.TransformerEncoderLayer(batch_first=True, **encoder_args)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        #Linear and dropout
        self.dropout = nn.Dropout(self.config.dropout)
        
        #2 layer approach: 
        hidden_dim = self.config.d_model
        self.mlp_1 = nn.Linear(hidden_dim, self.config.out_dim)
        self.mlp_2 = nn.Linear(self.config.out_dim, self.config.out_dim)
        
        #Loss functions 
        self.trip_margin_loss = nn.TripletMarginLoss(margin=1.0, reduction='mean')#p=2)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.pdist = nn.PairwiseDistance(p=2)
        
    def forward(self, x_i, src_mask, src_key_padding_mask):
        enc_out = self.encoder(x_i, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        lens = torch.logical_not(src_key_padding_mask).sum(dim=1).float()
        out = enc_out.sum(dim=1) / lens.unsqueeze(1)

        out = self.mlp_1(out)
        out = self.dropout(out)
        out = self.mlp_2(out)
        return out
    
    
    def triplet_margin_loss(self, output_seq1, output_seq2, output_seq3):
        loss = self.trip_margin_loss(output_seq1, output_seq2, output_seq3)
        return loss
    
    def distance_marginal_triplet(self, out_seq1, out_seq2, out_seq3, margin):
        d1 = self.pdist(out_seq1, out_seq2)
        d2 = self.pdist(out_seq1, out_seq3)
        zeros = torch.zeros(d1.shape).to(out_seq1)
        margin = margin.to(out_seq1)
        loss = torch.mean(torch.max(d1 - d2 + margin, zeros))

        return(loss)

    def distance_loss(self, output_seq1, output_seq2, output_seq3, margin):
        dist_seq1 = self.cos(output_seq1, output_seq2)
        dist_seq2 = self.cos(output_seq1, output_seq3)
        margin = margin.to(output_seq1)
        diff = dist_seq2 - dist_seq1
        dist_margin = self.l1_loss(diff.unsqueeze(0), margin.float().unsqueeze(0))
        
        return dist_margin

    def distance_loss2(self, output_seq1, output_seq2, output_seq3, margin):
        dist_seq1 = self.cos(output_seq1, output_seq2)
        dist_seq2 = self.cos(output_seq1, output_seq3)
        margin = margin.to(output_seq1)                                                                                                                 
        zeros = torch.zeros(dist_seq1.shape).to(output_seq1)
        loss = torch.mean(torch.max(dist_seq1 - dist_seq2 + margin, zeros))
        return loss
    
    def training_step(self, train_batch, batch_idx):
        margins = torch.FloatTensor(train_batch['key'])

        #Get the ID embeddings
        sequence_1 = train_batch['id']
        pad_mask_1 = train_batch['id_padding']
        sequence_2 = train_batch['positive']
        pad_mask_2 = train_batch['positive_padding']
        sequence_3 = train_batch['negative']
        pad_mask_3 = train_batch['negative_padding']
        
        out_seq1 = self.forward(sequence_1, src_mask=None, src_key_padding_mask=pad_mask_1)
        out_seq2 = self.forward(sequence_2, src_mask=None, src_key_padding_mask=pad_mask_2)
        out_seq3 = self.forward(sequence_3, src_mask=None, src_key_padding_mask=pad_mask_3)
        loss = self.distance_marginal_triplet(out_seq1, out_seq2, out_seq3, margins)
        
        self.log('train_loss', loss, sync_dist=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        margins = torch.FloatTensor(val_batch['key'])
        
        #Get the ID embeddings
        sequence_1 = val_batch['id']
        pad_mask_1 = val_batch['id_padding']
        sequence_2 = val_batch['positive']
        pad_mask_2 = val_batch['positive_padding']
        sequence_3 = val_batch['negative']
        pad_mask_3 = val_batch['negative_padding']

        out_seq1 = self.forward(sequence_1, src_mask=None, src_key_padding_mask=pad_mask_1)
        out_seq2 = self.forward(sequence_2, src_mask=None, src_key_padding_mask=pad_mask_2)
        out_seq3 = self.forward(sequence_3, src_mask=None, src_key_padding_mask=pad_mask_3)        
        loss = self.distance_marginal_triplet(out_seq1, out_seq2, out_seq3, margins)
        
        self.log('val_loss', loss, sync_dist=True)

        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr0)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        return [optimizer], [lr_scheduler]
