import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math
from typing import List, Callable, Union, Optional

from .base_regressor import BaseHead
from .super_mlp import AdptiveMLP, ModulatedMLP
from .harmonic_embedding import HarmonicEmbedding
from .hypernet import HypoNet, HyperNet

_logger = logging.getLogger(__name__)

class RushHead(BaseHead):
    def __init__(self, mean, use_homogeneous, in_channels, io_formula="BCHW", **kwargs) -> None:
        super().__init__(mean, use_homogeneous, in_channels, **kwargs)

        self.posEmbedder = HarmonicEmbedding(16)
        self.dataEmbedding=nn.Embedding(*[12,128], max_norm=1.0)

        num_pos = self.posEmbedder.get_output_dim(2)
        self.c_feature = 256
        self.m_net = ModulatedMLP(
            in_features=num_pos,
            out_features=4,
            z_length=self.c_feature,
            hidden_layers=[256,256],
        )
        self.c_net = HypoNet(
            in_features=in_channels,
            out_features=256,
            hidden_size=self.c_feature,
            num_hidden_layers=2,
            # outp_layer=True,
        )
        self.hyper_net = HyperNet(
            embedding_size=self.dataEmbedding.embedding_dim,
            hypo_arch=self.c_net,
            num_hidden_layers=1,
            hidden_dim=256,
        )
        # self.sc_reg = HyperCoNet(
        #     64, self.out_channels, [64,64], [64,64], io_formula, 8
        # )

        # self.sc_reg = nn.Sequential(
        #     nn.Conv2d(64, 64, 1),
        #     nn.ReLU(),
        #     # nn.Conv2d(64, 64, 1),
        #     # nn.ReLU(),
        #     nn.Conv2d(64, self.out_channels, 1)
        # )
        self.sc_reg = None

        self.sc_complement = nn.Sequential(
            nn.Linear(self.dataEmbedding.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def generate_feature_from_dataset_id(self, inputs, dataset_id_b1HW):
        B,C,H,W = inputs.shape
        features_repo = torch.empty(B,H,W, self.c_feature, device=inputs.device)
        for data_id in torch.unique(dataset_id_b1HW):
            data_embedding = self.dataEmbedding(data_id)
            net_params = self.hyper_net(data_embedding)
            features = self.c_net(
                inputs.permute(0, 2, 3, 1),
                params=net_params,
            )
            features_repo = torch.where(
                dataset_id_b1HW.permute(0,2,3,1)==data_id, features, features_repo)

        return features_repo
    
    def forward(self, inputs, norm_pos, dataset_ids, **kwargs):
        out_C = self.generate_feature_from_dataset_id(inputs, dataset_ids)

        pos_embedding = self.posEmbedder(norm_pos.permute(0,2,3,1)) # BxHxWxC
        out_M = self.m_net(
            z_code=out_C,
            pos_embedding=pos_embedding,
        )
        
        # old
        # out_M = F.relu(out_M)
        # sc = self.sc_reg(out_M.permute(0,3,1,2))

        sc = out_M.permute(0,3,1,2)

        B,C,H,W = inputs.shape
        data_embedding = self.dataEmbedding(dataset_ids.reshape(-1))
        data_embedding = data_embedding.reshape(B,H,W,-1)
        res_sc = self.sc_complement(data_embedding).permute(0,3,1,2)
        return self.refine_coordinates(sc)+res_sc
