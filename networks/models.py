#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import math

PI = math.pi

class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=[4],
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):

        if input.shape[1] > 3 and self.latent_dropout:
            x = F.dropout(input, p=0.2, training=self.training)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x

# Positional encoding like nerf
def positional_encoding(x, scale=None, l=10):
    ''' Implements positional encoding on the given coordinates.
    
    Differentiable wrt to x.
    
    Args:
        x: torch.Tensor(n, dim)  - input coordinates
        scale: torch.Tensor(2, dim) or None 
            scale along the coords for normalization
            If None - scale inferred from x
        l: int - number of modes in the encoding
    Returns:
        torch.Tensor(n, dim + 2 * dim * l) - positional encoded vector.
    '''

    if scale is None:
        scale = torch.vstack([x.min(axis=0)[0], x.max(axis=0)[0]]).T

    x_normed = 2 * (x - scale[:, 0]) / (scale[:, 1] - scale[:, 0]) - 1

    if l > 0:
        sinuses = torch.cat([torch.sin( (2 ** p) * PI * x_normed) for p in range(l) ], axis=1)
        cosines = torch.cat([torch.cos( (2 ** p) * PI * x_normed) for p in range(l) ], axis=1)

        pos_enc = torch.cat([x_normed, sinuses, cosines], axis=1)
    else:
        pos_enc = x_normed
    return pos_enc

class Scene_MLP(nn.Module):
    def __init__(self, latent_size, hidden_dims, do_pos_enc=False,pos_enc_freq=10,input_dim=3,norm_layers=(0,1,2,3), weight_norm=True,):
        super(Scene_MLP, self).__init__()
        
        self.norm_layers = norm_layers
        self.weight_norm = weight_norm
        self.do_pos_enc = do_pos_enc
        if do_pos_enc:
            input_dim *= (2*pos_enc_freq+1)
        layers = []
        dims = [input_dim] + hidden_dims + [latent_size]
        self.num_layers = len(dims)
        for layer in range(self.num_layers-1):
            if weight_norm and layer in self.norm_layers:
                layers.append(nn.utils.weight_norm(nn.Linear(dims[layer], dims[layer+1])))
            else:
                layers.append(nn.Linear(dims[layer], dims[layer+1]))

            if not weight_norm and self.norm_layers and layer in self.norm_layers:
                layers.append(nn.LayerNorm(dims[layer+1]))
        
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        if self.do_pos_enc:
            x = positional_encoding(x)
        x = self.layers(x)
        return x

# latent = torch.randn(10, 256)
# coords = torch.randn(10, 3)
# scene_model = Scene_MLP(256,[256])
# model = Decoder(256,[ 512, 512, 512, 512, 512, 512, 512, 512 ])
# print(model)
# lat = scene_model(coords)
# res = model(lat)
# print(res.shape)
# print(res)