import json
import inspect
from functools import partial
from dataclasses import dataclass, asdict
from model_protein_vec_single_variable import trans_basic_block_single, trans_basic_block_Config_single
from embed_structure_model import trans_basic_block_tmvec, trans_basic_block_Config_tmvec
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
    d_model: int = 512
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

        # Define 1D convolutional layer
        
        #2 layer approach: 
        self.mlp_1 = nn.Linear(self.config.d_model, self.config.out_dim)
        self.mlp_2 = nn.Linear(self.config.out_dim, self.config.out_dim)
        
        #Loss functions 
        self.trip_margin_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        
        #embedding lookup
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.l1_loss = nn.L1Loss(reduction='mean')
        
        self.pdist = nn.PairwiseDistance(p=2)

        ################## TM-Vec model
        vec_model_cpnt_tmvec = '/mnt/home/thamamsy/public_www/tm_vec_swiss_model_large.ckpt'
        vec_model_config_tmvec = '/mnt/home/thamamsy/public_www/tm_vec_swiss_model_large_params.json'
        
        #Load the model
        vec_model_config_tmvec = trans_basic_block_Config_tmvec.from_json(vec_model_config_tmvec)
        self.model_aspect_tmvec = trans_basic_block_tmvec.load_from_checkpoint(vec_model_cpnt_tmvec, config=vec_model_config_tmvec)
        for param in self.model_aspect_tmvec.parameters():
            param.requires_grad = False

        ################## PFam model
        vec_model_cpnt_pfam = '/mnt/home/thamamsy/ceph/protein_vec/models/model0.0001_dmodel1024_nlayer2_PFAM_negative_mining/checkpoints/epoch=0-step=37100-val_loss=0.0016.ckpt'
        vec_model_config_pfam = '/mnt/home/thamamsy/ceph/protein_vec/models/model0.0001_dmodel1024_nlayer2_PFAM_negative_mining/params.json'
        #Load the model
        vec_model_config_pfam = trans_basic_block_Config_single.from_json(vec_model_config_pfam)
        self.model_aspect_pfam = trans_basic_block_single.load_from_checkpoint(vec_model_cpnt_pfam, config=vec_model_config_pfam)
        for param in self.model_aspect_pfam.parameters():
            param.requires_grad = False

        ################## GENE3D model
        vec_model_cpnt_gene3D = '/mnt/home/thamamsy/ceph/protein_vec/models/model0.0001_dmodel1024_nlayer2_GENE3D_negative_mining/checkpoints/epoch=1-step=21960-val_loss=0.0021.ckpt'
        vec_model_config_gene3D = '/mnt/home/thamamsy/ceph/protein_vec/models/model0.0001_dmodel1024_nlayer2_GENE3D_negative_mining/params.json'
        #Load the model
        vec_model_config_gene3D = trans_basic_block_Config_single.from_json(vec_model_config_gene3D)
        self.model_aspect_gene3D = trans_basic_block_single.load_from_checkpoint(vec_model_cpnt_gene3D, config=vec_model_config_gene3D)
        for param in self.model_aspect_gene3D.parameters():
            param.requires_grad = False

        ################## EC model
        vec_model_cpnt_ec = '/mnt/home/thamamsy/ceph/protein_vec/models/model0.0001_dmodel1024_nlayer2_EC4_only_model_with_margins_4_8_12_25/checkpoints/last.ckpt'
        vec_model_config_ec = '/mnt/home/thamamsy/ceph/protein_vec/models/model0.0001_dmodel1024_nlayer2_EC4_only_model_with_margins_4_8_12_25/params.json'
        #Load the model
        vec_model_config_ec = trans_basic_block_Config_single.from_json(vec_model_config_ec)
        self.model_aspect_ec = trans_basic_block_single.load_from_checkpoint(vec_model_cpnt_ec, config=vec_model_config_ec)
        for param in self.model_aspect_ec.parameters():
            param.requires_grad = False

        ################## GO MFO model
        vec_model_cpnt_mfo = '/mnt/home/thamamsy/ceph/protein_vec/models/model0.0001_dmodel1024_nlayer4_GO_MFO_only_model/checkpoints/epoch=0-step=22704-val_loss=0.0233.ckpt'
        vec_model_config_mfo = '/mnt/home/thamamsy/ceph/protein_vec/models/model0.0001_dmodel1024_nlayer4_GO_MFO_only_model/params.json'
        #Load the model
        vec_model_config_mfo = trans_basic_block_Config_single.from_json(vec_model_config_mfo)
        self.model_aspect_mfo = trans_basic_block_single.load_from_checkpoint(vec_model_cpnt_mfo, config=vec_model_config_mfo)
        for param in self.model_aspect_mfo.parameters():
            param.requires_grad = False

        ################## GO BPO model
        vec_model_cpnt_bpo = '/mnt/home/thamamsy/ceph/protein_vec/models/model0.0001_dmodel1024_nlayer4_GO_BPO_only_model/checkpoints/epoch=0-step=21660-val_loss=0.0591.ckpt'
        vec_model_config_bpo = '/mnt/home/thamamsy/ceph/protein_vec/models/model0.0001_dmodel1024_nlayer4_GO_BPO_only_model/params.json'
        #Load the model 
        vec_model_config_bpo = trans_basic_block_Config_single.from_json(vec_model_config_bpo)
        self.model_aspect_bpo = trans_basic_block_single.load_from_checkpoint(vec_model_cpnt_bpo, config=vec_model_config_bpo)
        for param in self.model_aspect_bpo.parameters():
            param.requires_grad = False

        ################## GO CCO model
        vec_model_cpnt_cco = '/mnt/home/thamamsy/ceph/protein_vec/models/model0.0001_dmodel1024_nlayer4_GO_CCO_only_model/checkpoints/epoch=0-step=27324-val_loss=0.0170.ckpt'
        vec_model_config_cco = '/mnt/home/thamamsy/ceph/protein_vec/models/model0.0001_dmodel1024_nlayer4_GO_CCO_only_model/params.json'
        #Load the model
        vec_model_config_cco = trans_basic_block_Config_single.from_json(vec_model_config_cco)
        self.model_aspect_cco = trans_basic_block_single.load_from_checkpoint(vec_model_cpnt_cco, config=vec_model_config_cco)
        for param in self.model_aspect_cco.parameters():
            param.requires_grad = False

            
    def forward(self, x_i, src_key_padding_mask):
        #embedding
        src_key_padding_mask = src_key_padding_mask.to(x_i)
        enc_out = self.encoder(x_i, mask=None, src_key_padding_mask=src_key_padding_mask)
        lens = torch.logical_not(src_key_padding_mask).sum(dim=1).float()
        enc_out = enc_out.sum(dim=1) / lens.unsqueeze(1)
        out = self.mlp_1(enc_out)
        out = self.mlp_2(out)
        
        return out

    def distance_marginal_triplet(self, out_seq1, out_seq2, out_seq3, margin):
        d1 = self.pdist(out_seq1, out_seq2)
        d2 = self.pdist(out_seq1, out_seq3)
        zeros = torch.zeros(d1.shape).to(out_seq1)
        margin = margin.to(out_seq1)
        loss = torch.mean(torch.max(d1 - d2 + margin, zeros))
        return(loss)
    
    def triplet_margin_loss(self, output_seq1, output_seq2, output_seq3):
        loss = self.trip_margin_loss(output_seq1, output_seq2, output_seq3)
        return loss
    
    def distance_loss_tm_positive(self, output_seq1, output_seq2, tm_score):
        dist_seq = self.cos(output_seq1, output_seq2) 
        dist_tm = self.l1_loss(dist_seq.unsqueeze(0), tm_score.unsqueeze(0))
        return dist_tm

    def distance_loss_tm_difference(self, output_seq1, output_seq2, output_seq3, tm_score):
        dist_seq1 = self.cos(output_seq1, output_seq2)
        dist_seq2 = self.cos(output_seq1, output_seq3)
        difference = dist_seq2 - dist_seq1
        dist_tm = self.l1_loss(difference.unsqueeze(0), tm_score.unsqueeze(0))
        return dist_tm
    

    def make_matrix(self, sequence, pad_mask):
        pad_mask = pad_mask.to(sequence)
        aspect1 = self.model_aspect_tmvec(sequence, src_mask=None, src_key_padding_mask=pad_mask)[:,None,:]
        aspect2 = self.model_aspect_pfam(sequence, src_mask=None, src_key_padding_mask=pad_mask)[:,None,:]
        aspect3 = self.model_aspect_gene3D(sequence, src_mask=None, src_key_padding_mask=pad_mask)[:,None,:]
        aspect4 = self.model_aspect_ec(sequence, src_mask=None, src_key_padding_mask=pad_mask)[:,None,:]
        aspect5 = self.model_aspect_mfo(sequence, src_mask=None, src_key_padding_mask=pad_mask)[:,None,:]
        aspect6 = self.model_aspect_bpo(sequence, src_mask=None, src_key_padding_mask=pad_mask)[:,None,:]
        aspect7 = self.model_aspect_cco(sequence, src_mask=None, src_key_padding_mask=pad_mask)[:,None,:]
        combine_aspects = torch.cat([aspect1, aspect2, aspect3, aspect4, aspect5, aspect6, aspect7], dim=1)
        return combine_aspects
        
    def training_step(self, train_batch, batch_idx):
        lookup_dict = {
            'nothing': 1,  'ENZYME': 2, 'PFAM':3, 'MFO':4, 'BPO':5, 'CCO':6,
            'TM':7,'GENE3D':8
        }
        
        all_cols = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])
        
        sampled_keys = train_batch['key']
        margins = train_batch['margin']
        tm_type = train_batch['tm']
        tm_scores = train_batch['tm_scores']

        subset_sampled_keys = [sampled_keys[j].split(",") for j in range(len(sampled_keys))]
        masks = []
        for i in range(len(subset_sampled_keys)):
            mask = [all_cols[k] in subset_sampled_keys[i] for k in range(len(all_cols))]
            masks.append(mask)
        masks = torch.logical_not(torch.tensor(masks, dtype=torch.bool))
        

        #Get the ID embeddings
        sequence_1 = train_batch['id']
        pad_mask_1 = train_batch['id_padding']
        sequence_2 = train_batch['positive']
        pad_mask_2 = train_batch['positive_padding']
        sequence_3 = train_batch['negative']
        pad_mask_3 = train_batch['negative_padding']

        #Make Aspect matrices
        out_seq1 = self.make_matrix(sequence_1, pad_mask_1)
        out_seq2 = self.make_matrix(sequence_2, pad_mask_2)
        out_seq3 = self.make_matrix(sequence_3, pad_mask_3)

        #Forward pass
        out_seq1 = self.forward(out_seq1, masks)
        out_seq2 = self.forward(out_seq2, masks)
        out_seq3 = self.forward(out_seq3, masks)
                
        #Triplet loss
        loss_trip = self.distance_marginal_triplet(out_seq1, out_seq2, out_seq3, margins)

        #Positive TM loss
        loss_tm_positive = self.distance_loss_tm_positive(out_seq1, out_seq2, tm_scores)
        loss_positive_mask = torch.tensor([tm_type[i] == 'Positive' for i in range(len(tm_type))]).to(loss_tm_positive).to(bool)
        loss_tm_positive_fin = loss_tm_positive.masked_fill(loss_positive_mask, 0.0)

        #TM difference loss
        loss_tm_difference = self.distance_loss_tm_difference(out_seq1, out_seq2, out_seq3, tm_scores)
        loss_difference_mask = torch.tensor([tm_type[i] == 'Difference' for i in range(len(tm_type))]).to(loss_tm_difference).to(bool)
        loss_tm_difference_fin = loss_tm_difference.masked_fill(loss_difference_mask, 0.0)

        #Combined TM loss
        loss_part_2 = (loss_tm_positive_fin + loss_tm_difference_fin).mean()

        #complete loss
        loss = loss_trip + loss_part_2

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        all_cols = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])

        sampled_keys = val_batch['key']
        margins = val_batch['margin']
        tm_type = val_batch['tm']
        tm_scores = val_batch['tm_scores']

        subset_sampled_keys = [sampled_keys[j].split(",") for j in range(len(sampled_keys))]
        masks = []
        for i in range(len(subset_sampled_keys)):
            mask = [all_cols[k] in subset_sampled_keys[i] for k in range(len(all_cols))]
            masks.append(mask)
        masks = torch.logical_not(torch.tensor(masks, dtype=torch.bool))
        
        
        #Get the ID embeddings
        sequence_1 = val_batch['id']
        pad_mask_1 = val_batch['id_padding']
        sequence_2 = val_batch['positive']
        pad_mask_2 = val_batch['positive_padding']
        sequence_3 = val_batch['negative']
        pad_mask_3 = val_batch['negative_padding']

        out_seq1 = self.make_matrix(sequence_1, pad_mask_1)
        out_seq2 = self.make_matrix(sequence_2, pad_mask_2)
        out_seq3 = self.make_matrix(sequence_3, pad_mask_3)

        out_seq1 = self.forward(out_seq1, masks)
        out_seq2 = self.forward(out_seq2, masks)
        out_seq3 = self.forward(out_seq3, masks)

        #triplet loss
        loss_trip = self.distance_marginal_triplet(out_seq1, out_seq2, out_seq3, margins)

        #positive tm loss
        loss_tm_positive = self.distance_loss_tm_positive(out_seq1, out_seq2, tm_scores)
        loss_positive_mask = torch.tensor([tm_type[i] == 'Positive' for i in range(len(tm_type))]).to(loss_tm_positive).to(bool)
        loss_tm_positive_fin = loss_tm_positive.masked_fill(loss_positive_mask, 0.0)

        #difference tm loss
        loss_tm_difference = self.distance_loss_tm_difference(out_seq1, out_seq2, out_seq3, tm_scores)
        loss_difference_mask = torch.tensor([tm_type[i] == 'Difference' for i in range(len(tm_type))]).to(loss_tm_difference).to(bool)
        loss_tm_difference_fin = loss_tm_difference.masked_fill(loss_difference_mask, 0.0)

        #complete loss
        loss_part_2 = (loss_tm_positive_fin + loss_tm_difference_fin).mean()
        #complete loss
        loss = loss_trip + loss_part_2
        
        self.log('val_loss', loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr0)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
        return [optimizer], [lr_scheduler]
