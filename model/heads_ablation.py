import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math
from typing import List, Callable, Union, Optional

from .heads import RushHead
from .super_mlp import AdptiveMLP, ModulatedMLP

class RH_no_hypernet(RushHead):
    def __init__(self, mean, use_homogeneous, in_channels, io_formula="BCHW", **kwargs) -> None:
        super().__init__(mean, use_homogeneous, in_channels, io_formula, **kwargs)

        self.c_net = nn.Sequential(
            nn.Linear(in_channels, self.c_feature),
            nn.ReLU(),
            nn.Linear(self.c_feature, 256),
            nn.ReLU(),
            # keep same to the original hyponet
            # in_features=in_channels,
            # out_features=256,
            # hidden_size=self.c_feature,
            # num_hidden_layers=2,
            # outp_layer=True,
        )
        self.hyper_net = None

    def generate_feature_from_dataset_id(self, inputs, dataset_id_b1HW):
        features = self.c_net(
                inputs.permute(0, 2, 3, 1),
        )
        return features
    
class RH_no_modulation(RushHead):
    def __init__(self, mean, use_homogeneous, in_channels, io_formula="BCHW", **kwargs) -> None:
        super().__init__(mean, use_homogeneous, in_channels, io_formula, **kwargs)

        num_pos = self.posEmbedder.get_output_dim(2)
        self.m_net = nn.ModuleList(
            [nn.Linear(num_pos+self.c_feature, 256),
            nn.Linear(256+self.c_feature, 256),
            nn.Linear(256+self.c_feature, 4),]
            # in_features=num_pos,
            # out_features=64,
            # z_length=self.c_feature,
            # hidden_layers=[256,256],
        )

    def forward(self, inputs, norm_pos, dataset_ids, **kwargs):
        out_C = self.generate_feature_from_dataset_id(inputs, dataset_ids)

        pos_embedding = self.posEmbedder(norm_pos.permute(0,2,3,1)) # BxHxWxC

        m_out = pos_embedding
        for layer in self.m_net[:-1]:
            m_out = layer(torch.concat([m_out, out_C], dim=-1))
            m_out = F.relu(m_out)
        sc = self.m_net[-1](torch.concat([m_out, out_C], dim=-1)).permute(0,3,1,2)
        # out_M = self.m_net(
        #     z_code=out_C,
        #     pos_embedding=pos_embedding,
        # )
        # out_M = F.relu(out_M)

        # sc = self.sc_reg(out_M.permute(0,3,1,2))

        B,C,H,W = inputs.shape
        data_embedding = self.dataEmbedding(dataset_ids.reshape(-1))
        data_embedding = data_embedding.reshape(B,H,W,-1)
        res_sc = self.sc_complement(data_embedding).permute(0,3,1,2)
        return self.refine_coordinates(sc)+res_sc
    
class RH_no_pos(RushHead):
    def __init__(self, mean, use_homogeneous, in_channels, io_formula="BCHW", **kwargs) -> None:
        super().__init__(mean, use_homogeneous, in_channels, io_formula, **kwargs)

        self.posEmbedder = nn.Identity()
        num_pos = 2
        self.c_feature = 256
        self.m_net = ModulatedMLP(
            in_features=num_pos,
            out_features=4,
            z_length=self.c_feature,
            hidden_layers=[256,256],
        )

class zero_out(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.zeros([1,1,1,1], device=input.device)
    
class RH_no_complement(RushHead):
    def __init__(self, mean, use_homogeneous, in_channels, io_formula="BCHW", **kwargs) -> None:
        super().__init__(mean, use_homogeneous, in_channels, io_formula, **kwargs)

        self.sc_complement = zero_out()

class RH_no_reg(RushHead):
    def __init__(self, mean, use_homogeneous, in_channels, io_formula="BCHW", **kwargs) -> None:
        super().__init__(mean, use_homogeneous, in_channels, io_formula, **kwargs)
        self.sc_reg = None

        num_pos = self.posEmbedder.get_output_dim(2)
        self.m_net = ModulatedMLP(
            in_features=num_pos,
            out_features=4,
            z_length=self.c_feature,
            hidden_layers=[256,256],
        )

    def forward(self, inputs, norm_pos, dataset_ids, **kwargs):
        out_C = self.generate_feature_from_dataset_id(inputs, dataset_ids)

        pos_embedding = self.posEmbedder(norm_pos.permute(0,2,3,1)) # BxHxWxC
        sc = self.m_net(
            z_code=out_C,
            pos_embedding=pos_embedding,
        ).permute(0,3,1,2)

        B,C,H,W = inputs.shape
        data_embedding = self.dataEmbedding(dataset_ids.reshape(-1))
        data_embedding = data_embedding.reshape(B,H,W,-1)
        res_sc = self.sc_complement(data_embedding).permute(0,3,1,2)
        return self.refine_coordinates(sc)+res_sc