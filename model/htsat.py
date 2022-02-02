# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# Model Core
# below codes are based and referred from https://github.com/microsoft/Swin-Transformer
# Swin Transformer for Computer Vision: https://arxiv.org/pdf/2103.14030.pdf


import logging
import pdb
import math
import random
from numpy.core.fromnumeric import clip, reshape
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from itertools import repeat
from typing import List
from .layers import PatchEmbed, Mlp, DropPath, trunc_normal_, to_2tuple
from utils import do_mixup, interpolate


# below codes are based and referred from https://github.com/microsoft/Swin-Transformer
# Swin Transformer for Computer Vision: https://arxiv.org/pdf/2103.14030.pdf

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


# We use the model based on Swintransformer Block, therefore we can use the swin-transformer pretrained model
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_before_mlp='ln'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm_before_mlp = norm_before_mlp
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.norm_before_mlp == 'ln':
            self.norm2 = nn.LayerNorm(dim)
        elif self.norm_before_mlp == 'bn':
            self.norm2 = lambda x: nn.BatchNorm1d(dim)(x.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # pdb.set_trace()
        H, W = self.input_resolution
        # print("H: ", H)
        # print("W: ", W)
        # pdb.set_trace()
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows, attn = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 norm_before_mlp='ln'):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, norm_before_mlp=norm_before_mlp)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        attns = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, attn = blk(x)
                if not self.training:
                    attns.append(attn.unsqueeze(0))
        if self.downsample is not None:
            x = self.downsample(x)
        if not self.training:
            attn = torch.cat(attns, dim = 0)
            attn = torch.mean(attn, dim = 0)
        return x, attn

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


# The Core of HTSAT
class HTSAT_Swin_Transformer(nn.Module):
    r"""HTSAT based on the Swin Transformer
    Args:
        spec_size (int | tuple(int)): Input Spectrogram size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 4
        path_stride (iot | tuple(int)): Patch Stride for Frequency and Time Axis. Default: 4
        in_chans (int): Number of input image channels. Default: 1 (mono)
        num_classes (int): Number of classes for classification head. Default: 527
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each HTSAT-Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        config (module): The configuration Module from config.py
    """

    def __init__(self, spec_size=256, patch_size=4, patch_stride=(4,4), 
                in_chans=1, num_classes=527,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, 
                 ape=False, patch_norm=True,
                 use_checkpoint=False, norm_before_mlp='ln', config = None, **kwargs):
        super(HTSAT_Swin_Transformer, self).__init__()

        self.config = config
        self.spec_size = spec_size 
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.ape = ape
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = len(self.depths)
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))
        
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        self.qkv_bias = qkv_bias
        self.qk_scale = None

        self.patch_norm = patch_norm
        self.norm_layer = norm_layer if self.patch_norm else None
        self.norm_before_mlp = norm_before_mlp
        self.mlp_ratio = mlp_ratio

        self.use_checkpoint = use_checkpoint

        #  process mel-spec ; used only once
        self.freq_ratio = self.spec_size // self.config.mel_bins
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 32     # Downsampled ratio
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=config.window_size, hop_length=config.hop_size, 
            win_length=config.window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=config.sample_rate, n_fft=config.window_size, 
            n_mels=config.mel_bins, fmin=config.fmin, fmax=config.fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2) # 2 2
        self.bn0 = nn.BatchNorm2d(self.config.mel_bins)


        # split spctrogram into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=self.spec_size, patch_size=self.patch_size, in_chans=self.in_chans, 
            embed_dim=self.embed_dim, norm_layer=self.norm_layer, patch_stride = patch_stride)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                    patches_resolution[1] // (2 ** i_layer)),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=self.norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                norm_before_mlp=self.norm_before_mlp)
            self.layers.append(layer)

        # A deprecated optimization for using a hierarchical output from different blocks
        # if self.config.htsat_hier_output:
        #     self.norm = nn.ModuleList(
        #         [self.norm_layer(
        #             min(
        #               self.embed_dim * (2 ** (len(self.depths) - 1)),
        #               self.embed_dim * (2 ** (i + 1)) 
        #                 )
        #         ) for i in range(len(self.depths))]   
        #     )
        # else:

        self.norm = self.norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        
        # A deprecated optimization for using the max value instead of average value
        # if self.config.htsat_use_max:
        #     self.a_avgpool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        #     self.a_maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        if self.config.enable_tscam:
            # if self.config.htsat_hier_output:
            #     self.tscam_conv = nn.ModuleList()
            #     for i in range(len(self.depths)):
            #         zoom_ratio = 2 ** min(len(self.depths) - 1, i + 1)
            #         zoom_dim = min(
            #             self.embed_dim * (2 ** (len(self.depths) - 1)),
            #             self.embed_dim * (2 ** (i + 1)) 
            #         )
            #         SF = self.spec_size // zoom_ratio // self.patch_stride[0] // self.freq_ratio
            #         self.tscam_conv.append(
            #             nn.Conv2d(
            #                 in_channels = zoom_dim,
            #                 out_channels = self.num_classes,
            #                 kernel_size = (SF, 3),
            #                 padding = (0,1)
            #             )
            #         )
            #     self.head = nn.Linear(num_classes * len(self.depths), num_classes)
            # else:

            SF = self.spec_size // (2 ** (len(self.depths) - 1)) // self.patch_stride[0] // self.freq_ratio
            self.tscam_conv = nn.Conv2d(
                in_channels = self.num_features,
                out_channels = self.num_classes,
                kernel_size = (SF,3),
                padding = (0,1)
            )
            self.head = nn.Linear(num_classes, num_classes)
        else:
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


    def forward_features(self, x):
        # A deprecated optimization for using a hierarchical output from different blocks
        # if self.config.htsat_hier_output:
        #     hier_x = []
        #     hier_attn = []

        frames_num = x.shape[2]        
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            x, attn = layer(x)
            # A deprecated optimization for using a hierarchical output from different blocks
            # if self.config.htsat_hier_output:
            #     hier_x.append(x)
            #     if i == len(self.layers) - 1:
            #         hier_attn.append(attn)

        # A deprecated optimization for using a hierarchical output from different blocks
        # if self.config.htsat_hier_output:
        #     hxs = []
        #     fphxs = []
        #     for i in range(len(hier_x)):
        #         hx = hier_x[i]
        #         hx = self.norm[i](hx)
        #         B, N, C = hx.shape
        #         zoom_ratio = 2 ** min(len(self.depths) - 1, i + 1)
        #         SF = frames_num // zoom_ratio // self.patch_stride[0]
        #         ST = frames_num // zoom_ratio // self.patch_stride[1]
        #         hx = hx.permute(0,2,1).contiguous().reshape(B, C, SF, ST)
        #         B, C, F, T = hx.shape
        #         c_freq_bin = F // self.freq_ratio
        #         hx = hx.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
        #         hx = hx.permute(0,1,3,2,4).contiguous().reshape(B, C, c_freq_bin, -1)
                
        #         hx = self.tscam_conv[i](hx)
        #         hx = torch.flatten(hx, 2)
        #         fphx = interpolate(hx.permute(0,2,1).contiguous(), self.spec_size * self.freq_ratio // hx.shape[2])
                
        #         hx = self.avgpool(hx)
        #         hx = torch.flatten(hx, 1)
        #         hxs.append(hx)
        #         fphxs.append(fphx)
        #     hxs = torch.cat(hxs, dim=1)
        #     fphxs = torch.cat(fphxs, dim = 2)
        #     hxs = self.head(hxs)
        #     fphxs = self.head(fphxs)
        #     output_dict = {'framewise_output': torch.sigmoid(fphxs), 
        #         'clipwise_output': torch.sigmoid(hxs)}
        #     return output_dict

        if self.config.enable_tscam:
            # for x
            x = self.norm(x)
            B, N, C = x.shape
            SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
            ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
            x = x.permute(0,2,1).contiguous().reshape(B, C, SF, ST)
            B, C, F, T = x.shape
            # group 2D CNN
            c_freq_bin = F // self.freq_ratio
            x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
            x = x.permute(0,1,3,2,4).contiguous().reshape(B, C, c_freq_bin, -1)

            # get latent_output
            latent_output = self.avgpool(torch.flatten(x,2))
            latent_output = torch.flatten(latent_output, 1)

            # display the attention map, if needed
            if self.config.htsat_attn_heatmap:
                # for attn
                attn = torch.mean(attn, dim = 1)
                attn = torch.mean(attn, dim = 1)
                attn = attn.reshape(B, SF, ST)
                c_freq_bin = SF // self.freq_ratio
                attn = attn.reshape(B, SF // c_freq_bin, c_freq_bin, ST) 
                attn = attn.permute(0,2,1,3).contiguous().reshape(B, c_freq_bin, -1)
                attn = attn.mean(dim = 1)
                attn_max = torch.max(attn, dim = 1, keepdim = True)[0]
                attn_min = torch.min(attn, dim = 1, keepdim = True)[0]
                attn = ((attn * 0.15) + (attn_max * 0.85 - attn_min)) / (attn_max - attn_min)
                attn = attn.unsqueeze(dim = 2)

            x = self.tscam_conv(x)
            x = torch.flatten(x, 2) # B, C, T
            
            # A deprecated optimization for using the max value instead of average value
            # if self.config.htsat_use_max:
            #     x1 = self.a_maxpool(x)
            #     x2 = self.a_avgpool(x)
            #     x = x1 + x2

            if self.config.htsat_attn_heatmap:
                fpx = interpolate(torch.sigmoid(x).permute(0,2,1).contiguous() * attn, 8 * self.patch_stride[1]) 
            else: 
                fpx = interpolate(torch.sigmoid(x).permute(0,2,1).contiguous(), 8 * self.patch_stride[1]) 
            
            # A deprecated optimization for using the max value instead of average value
            # if self.config.htsat_use_max:
            #     x1 = self.avgpool(x)
            #     x2 = self.maxpool(x)
            #     x = x1 + x2
            # else:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)

            if self.config.loss_type == "clip_ce":
                output_dict = {
                    'framewise_output': fpx, # already sigmoided
                    'clipwise_output': x,
                    'latent_output': latent_output
                }
            else:
                output_dict = {
                    'framewise_output': fpx, # already sigmoided
                    'clipwise_output': torch.sigmoid(x),
                    'latent_output': latent_output
                }
           
        else:
            x = self.norm(x)  # B N C
            B, N, C = x.shape
            
            fpx = x.permute(0,2,1).contiguous().reshape(B, C, frames_num // (2 ** (len(self.depths) + 1)), frames_num // (2 ** (len(self.depths) + 1)) )
            B, C, F, T = fpx.shape
            c_freq_bin = F // self.freq_ratio
            fpx = fpx.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
            fpx = fpx.permute(0,1,3,2,4).contiguous().reshape(B, C, c_freq_bin, -1)
            fpx = torch.sum(fpx, dim = 2)
            fpx = interpolate(fpx.permute(0,2,1).contiguous(), 8 * self.patch_stride[1]) 
            x = self.avgpool(x.transpose(1, 2))  # B C 1
            x = torch.flatten(x, 1)
            if self.num_classes > 0:
                x = self.head(x)
                fpx = self.head(fpx)
            output_dict = {'framewise_output': torch.sigmoid(fpx), 
                'clipwise_output': torch.sigmoid(x)}
        return output_dict

    def crop_wav(self, x, crop_size, spe_pos = None):
        time_steps = x.shape[2]
        tx = torch.zeros(x.shape[0], x.shape[1], crop_size, x.shape[3]).to(x.device)
        for i in range(len(x)):
            if spe_pos is None:
                crop_pos = random.randint(0, time_steps - crop_size - 1)
            else:
                crop_pos = spe_pos
            tx[i][0] = x[i, 0, crop_pos:crop_pos + crop_size,:]
        return tx

    # Reshape the wavform to a img size, if you want to use the pretrained swin transformer model
    def reshape_wav2img(self, x):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert T <= target_T and F <= target_F, "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode="bicubic", align_corners=True)
        if F < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode="bicubic", align_corners=True)
        x = x.permute(0,1,3,2).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.freq_ratio, x.shape[3] // self.freq_ratio)
        # print(x.shape)
        x = x.permute(0,1,3,2,4).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
        return x
    
    # Repeat the wavform to a img size, if you want to use the pretrained swin transformer model
    def repeat_wat2img(self, x, cur_pos):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert T <= target_T and F <= target_F, "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode="bicubic", align_corners=True)
        if F < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode="bicubic", align_corners=True)  
        x = x.permute(0,1,3,2).contiguous() # B C F T
        x = x[:,:,:,cur_pos:cur_pos + self.spec_size]
        x = x.repeat(repeats = (1,1,4,1))
        return x

    def forward(self, x: torch.Tensor, mixup_lambda = None, infer_mode = False):# out_feat_keys: List[str] = None):
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        if infer_mode:
            # in infer mode. we need to handle different length audio input
            frame_num = x.shape[2]
            target_T = int(self.spec_size * self.freq_ratio)
            repeat_ratio = math.floor(target_T / frame_num)
            x = x.repeat(repeats=(1,1,repeat_ratio,1))
            x = self.reshape_wav2img(x)
            output_dict = self.forward_features(x)
        elif self.config.enable_repeat_mode:
            if self.training:
                cur_pos = random.randint(0, (self.freq_ratio - 1) * self.spec_size - 1)
                x = self.repeat_wat2img(x, cur_pos)
                output_dict = self.forward_features(x)
            else:
                output_dicts = []
                for cur_pos in range(0, (self.freq_ratio - 1) * self.spec_size + 1, self.spec_size):
                    tx = x.clone()
                    tx = self.repeat_wat2img(tx, cur_pos)
                    output_dicts.append(self.forward_features(tx))
                clipwise_output = torch.zeros_like(output_dicts[0]["clipwise_output"]).float().to(x.device)
                framewise_output = torch.zeros_like(output_dicts[0]["framewise_output"]).float().to(x.device)
                for d in output_dicts:
                    clipwise_output += d["clipwise_output"]
                    framewise_output += d["framewise_output"]
                clipwise_output  = clipwise_output / len(output_dicts)
                framewise_output = framewise_output / len(output_dicts)

                output_dict = {
                    'framewise_output': framewise_output, 
                    'clipwise_output': clipwise_output
                }
        else:
            if x.shape[2] > self.freq_ratio * self.spec_size:
                if self.training:
                    x = self.crop_wav(x, crop_size=self.freq_ratio * self.spec_size)
                    x = self.reshape_wav2img(x)
                    output_dict = self.forward_features(x)
                else:
                    # Change: Hard code here
                    overlap_size = (x.shape[2] - 1) // 4
                    output_dicts = []
                    crop_size = (x.shape[2] - 1) // 2
                    for cur_pos in range(0, x.shape[2] - crop_size - 1, overlap_size):
                        tx = self.crop_wav(x, crop_size = crop_size, spe_pos = cur_pos)
                        tx = self.reshape_wav2img(tx)
                        output_dicts.append(self.forward_features(tx))
                    clipwise_output = torch.zeros_like(output_dicts[0]["clipwise_output"]).float().to(x.device)
                    framewise_output = torch.zeros_like(output_dicts[0]["framewise_output"]).float().to(x.device)
                    for d in output_dicts:
                        clipwise_output += d["clipwise_output"]
                        framewise_output += d["framewise_output"]
                    clipwise_output  = clipwise_output / len(output_dicts)
                    framewise_output = framewise_output / len(output_dicts)
                    output_dict = {
                        'framewise_output': framewise_output, 
                        'clipwise_output': clipwise_output
                    }
            else: # this part is typically used, and most easy one
                x = self.reshape_wav2img(x)
                output_dict = self.forward_features(x)
        # x = self.head(x)
        return output_dict