import re
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Union, Optional

from .acenet import Encoder

_logger = logging.getLogger(__name__)

class BaseHead(nn.Module):
    def __init__(self, 
                 mean,
                 use_homogeneous,
                 in_channels,
                 **kwargs
                 ) -> None:
        super().__init__()
        # Learn scene coordinates relative to a mean coordinate (e.g. center of the scene).
        self.register_buffer("mean", mean.clone().detach().view(1, 3, 1, 1))
        
        self.use_homogeneous = use_homogeneous

        if self.use_homogeneous:
            out_channels = 4
            homogeneous_min_scale=0.01
            homogeneous_max_scale=4.0
            # Use buffers because they need to be saved in the state dict.
            self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
            self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
            self.register_buffer("max_inv_scale", 1. / self.max_scale)
            self.register_buffer("h_beta", math.log(2) / (1. - self.max_inv_scale))
            self.register_buffer("min_inv_scale", 1. / self.min_scale)
        else:
            out_channels = 3
        self.register_buffer('out_channels', torch.tensor(out_channels))
        self.sc_reg = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, inputs, **kwargs):
        """
        Forward pass.
        """
        sc = self.get_scs(inputs)
        return self.refine_coordinates(sc)
    
    def get_scs(self, features):
        return self.sc_reg(features)

    def refine_coordinates(self, sc):
        if self.use_homogeneous:
            # Dehomogenize coords:
            # Softplus ensures we have a smooth homogeneous parameter with a minimum value = self.max_inv_scale.
            h_slice = F.softplus(sc[:, 3, :, :].unsqueeze(1), beta=self.h_beta.item()) + self.max_inv_scale
            h_slice.clamp_(max=self.min_inv_scale)
            sc = sc[:, :3] / h_slice

        # Add the mean to the predicted coordinates.
        sc += self.mean
        return sc

class BaseRegressor(nn.Module):
    OUTPUT_SUBSAMPLE = 8
    def __init__(
        self,
            mean,
            use_homogeneous,
            num_encoder_features=512,
            Head: nn.Module = BaseHead,
            **kwargs
    ):
        super().__init__()

        self.feature_dim = num_encoder_features
        self.encoder = Encoder(out_channels=self.feature_dim, pool=False)
        self.heads = Head(mean, use_homogeneous, in_channels=self.feature_dim)

    @classmethod
    def create_from_encoder(
        cls, encoder_state_dict, mean, use_homogeneous, **kwargs):
        """
        Create a regressor using a pretrained encoder, loading encoder-specific parameters from the state dict.

        encoder_state_dict: pretrained encoder state dictionary.
        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        """

        # Number of output channels of the last encoder layer.
        num_encoder_features = encoder_state_dict['res2_conv3.weight'].shape[0]

        # Create a regressor.
        _logger.info(f"Creating Regressor using pretrained encoder with {num_encoder_features} feature size.")
        regressor = cls(
            mean=mean,
            use_homogeneous=use_homogeneous,
            num_encoder_features=num_encoder_features,
            **kwargs
        )

        # Load encoder weights.
        regressor.encoder.load_state_dict(encoder_state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_state_dict(cls, state_dict, use_homogeneous, **kwargs):
        """
        Instantiate a regressor from a pretrained state dictionary.

        state_dict: pretrained state dictionary.
        """
        # Mean is zero (will be loaded from the state dict).
        mean = torch.zeros((3,))

        # Number of output channels of the last encoder layer.
        num_encoder_features = state_dict['encoder.res2_conv3.weight'].shape[0]

        # Create a regressor.
        _logger.info(f"Creating regressor from pretrained state_dict:"
                     f"\n\tHomogeneous coordinates: {use_homogeneous}"
                     f"\n\tEncoder feature size: {num_encoder_features}")
        regressor = cls(
            mean, 
            use_homogeneous,
            num_encoder_features,
            **kwargs
        )

        # Load all weights.
        regressor.load_state_dict(state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_split_state_dict(cls, encoder_state_dict, head_state_dict, **kwargs):
        """
        Instantiate a regressor from a pretrained encoder (scene-agnostic) and a scene-specific head.

        encoder_state_dict: encoder state dictionary
        head_state_dict: scene-specific head state dictionary
        """
        # We simply merge the dictionaries and call the other constructor.
        merged_state_dict = {}

        for k, v in encoder_state_dict.items():
            merged_state_dict[f"encoder.{k}"] = v

        for k, v in head_state_dict.items():
            merged_state_dict[f"heads.{k}"] = v

        use_homogeneous = True if head_state_dict['out_channels'] == 4 else False
        return cls.create_from_state_dict(merged_state_dict, use_homogeneous, **kwargs)

    def load_encoder(self, encoder_dict_file):
        """
        Load weights into the encoder network.
        """
        self.encoder.load_state_dict(torch.load(encoder_dict_file))

    def get_features(self, inputs):
        return self.encoder(inputs)

    def get_scene_coordinates(self, features, **kwargs):
        return self.heads(features, **kwargs)

    def forward(self, inputs, **kwargs):
        """
        Forward pass.
        """
        features = self.get_features(inputs)
        return self.get_scene_coordinates(features, **kwargs)