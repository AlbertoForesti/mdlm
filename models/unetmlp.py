
from functools import partial
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from pytorch_lightning import LightningModule
import torch
import math

from torch import nn


# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(sigma, *args, **kwargs):
    return sigma


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# small helper modules

import torch.nn as nn

class Permute(LightningModule):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class Unpermute(LightningModule):
    def __init__(self, *dims):
        super(Unpermute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class Residual(LightningModule):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        # nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Linear(dim, default(dim_out, dim))
    )


def Downsample(dim, dim_out=None):
    return nn.Linear(dim, default(dim_out, dim))


class Residual(LightningModule):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Block(LightningModule):
    def __init__(self, dim, dim_out, groups=8, shift_scale=True):
        super().__init__()
        # self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.proj = nn.Linear(dim, dim_out)
        self.act = nn.SiLU()
        self.permute = Permute(0, 2, 1)
        self.unpermute = Unpermute(0, 2, 1)
        # self.act = nn.Relu()
        self.norm = nn.GroupNorm(groups, dim)
        self.dim = dim
        # self.norm = nn.BatchNorm1d( dim)
        self.shift_scale = shift_scale

    def forward(self, x, sigma=None):
        x = self.permute(x)
        x = self.norm(x)
        x = self.unpermute(x)
        x = self.act(x)
        x = self.proj(x)

        if exists(sigma):
            if self.shift_scale:
                scale, shift = sigma
                x = x * (scale.squeeze() + 1) + shift.squeeze()
            else:
                try:
                    sigma = sigma.unsqueeze(1).expand_as(x)
                except:
                    raise UserWarning(f"Incompatible shapes: x={x.shape}, sigma={sigma.shape}")
                try:
                    x = x + sigma
                except:
                    raise UserWarning(f"Incompatible shapes: x={x.shape}, sigma={sigma.shape}")
        return x


class ResnetBlock(LightningModule):
    def __init__(self, dim, dim_out, *, sigma_emb_dim=None, groups=32, shift_scale=False, scaling_factor=None):
        super().__init__()
        self.shift_scale = shift_scale
        self.mlp = nn.Sequential(
            nn.SiLU(),
            # nn.Linear(sigma_emb_dim, dim_out * 2)
            nn.Linear(sigma_emb_dim, dim_out*2 if shift_scale else dim_out)
        ) if exists(sigma_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups,
                            shift_scale=shift_scale)
        self.block2 = Block(dim_out, dim_out, groups=groups,
                            shift_scale=shift_scale)
        # self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.lin_layer = nn.Linear(
            dim, dim_out) if dim != dim_out else nn.Identity()
        
        if scaling_factor is None:
            self.scaling_factor = 1.0
        else:
            self.scaling_factor = 2**(-scaling_factor/2)

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):

            time_emb = self.mlp(time_emb)
            scale_shift = time_emb

        h = self.block1(x, sigma=scale_shift)

        h = self.block2(h)

        out = h + self.lin_layer(x)*self.scaling_factor

        return out

class EmbeddingLayer(LightningModule):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors, 
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class UnetMLP(LightningModule):
    def __init__(
        self,
        config,
        vocab_size
    ):
        super().__init__()

        # determine dimensions
        self.vocab_size = vocab_size
        self.absorb = "absorb"
        self.config = config
        self.resnet_block_groups = config.model.resnet_block_groups
        self.sigma_dim = config.model.sigma_dim
        self.dim_mults = config.model.dim_mults

        self.is_parametric_marginal = config.model.is_parametric_marginal
        self.use_marginal_flag = config.model.use_marginal_flag
        init_dim = config.model.init_dim
        if init_dim == None:
            init_dim = (self.sequence_length + 1) * self.dim_mults[0]

        dims = [init_dim, *map(lambda m: init_dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=self.resnet_block_groups)

        self.embedding_layer = EmbeddingLayer(init_dim, self.vocab_size)

        if self.is_parametric_marginal and self.use_marginal_flag:
            sigma_init_dim = 2
        else:
            sigma_init_dim = 1

        self.sigma_mlp = nn.Sequential(
            nn.Linear(sigma_init_dim, self.sigma_dim),
            nn.GELU(),
            nn.Linear(self.sigma_dim, self.sigma_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            if config.model.scaling_factor is None:
                scaling_factor = None
            else:
                scaling_factor = config.model.scaling_factor * ind

            module = nn.ModuleList([block_klass(dim_in, dim_in, sigma_emb_dim=self.sigma_dim, scaling_factor=scaling_factor),
                                    #        block_klass(dim_in, dim_in, sigma_emb_dim = sigma_dim)
                                    ])

            # module.append( Downsample(dim_in, dim_out) if not is_last else nn.Linear(dim_in, dim_out))
            self.downs.append(module)

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, sigma_emb_dim=self.sigma_dim)

        # self.mid_block2 = block_klass(joint_dim, mid_dim, sigma_emb_dim = sigma_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            module = nn.ModuleList([block_klass(dim_out + dim_in, dim_out, sigma_emb_dim=self.sigma_dim),
                                    #       block_klass(dim_out + dim_in, dim_out, sigma_emb_dim = sigma_dim)
                                    ])
            # module.append( Upsample(dim_out, dim_in) if not is_last else  nn.Linear(dim_out, dim_in))
            self.ups.append(module)

        # default_out_dim = channels * (1 if not learned_variance else 2)

        self.out_dim = dim_in

        self.final_res_block = block_klass(
            init_dim * 2, init_dim, sigma_emb_dim=self.sigma_dim)
    
        self.proj = nn.Linear(init_dim, self.vocab_size)
        self.p_enc_1d_model_sum = Summer(PositionalEncoding1D(init_dim))

        # self.proj.weight.data.fill_(0.0)
        # self.proj.bias.data.fill_(0.0)

        self.final_lin = nn.Sequential(
            Permute(0, 2, 1),  # Change shape to (batch_size, channels, token_dim)
            nn.GroupNorm(self.resnet_block_groups, init_dim),
            Unpermute(0, 2, 1),  # Change shape back to (batch_size, token_dim, channels)
            nn.SiLU(),
            self.proj
        )

    def forward(self, indices, sigma, marginal_flag=None, std=None):
        assert sigma.ndim == 2, f"Expected two dimensions of sigma, got sigma.ndim={sigma.ndim}"
        x = self.embedding_layer(indices)
        x = self.p_enc_1d_model_sum(x)

        r = x.clone()

        if self.is_parametric_marginal and self.use_marginal_flag:
            assert marginal_flag is not None, "marginal_flag must be provided if is_parametric_marginal is True"
            marginal_flag = torch.tensor(marginal_flag).float().to(x.device)
            marginal_flag = marginal_flag.expand(x.size(0), 1)
            try:
                sigma = torch.cat((sigma, marginal_flag), dim=-1)
            except:
                raise UserWarning(f"Devices: sigma={sigma.device}, marginal_flag={marginal_flag.device}, x={x.device}, indices={indices.device}")

        sigma = self.sigma_mlp(sigma)

        h = []

        for blocks in self.downs:

            block1 = blocks[0]

            x = block1(x, sigma)

            h.append(x)
        #   x = downsample(x)

        # x = self.mid_block1(x, sigma)

        # x = self.mid_block2(x, sigma)

        for blocks in self.ups:

            block1 = blocks[0]
            x = torch.cat((x, h.pop()), dim=-1)
            x = block1(x, sigma)

            # x = torch.cat((x, h.pop()), dim = 1)
            # x = block2(x, sigma)

           # x = upsample(x)

        x = torch.cat((x, r), dim=-1)

        x = self.final_res_block(x, sigma)

        if self.config.parameterization == "mine":
            return x

        if std != None:
            x = self.final_lin(x) / std
        else:
            x = self.final_lin(x)
        
        # x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
        
        return x
