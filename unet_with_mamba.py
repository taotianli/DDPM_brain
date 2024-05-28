import math

from einops import rearrange
from torch import Tensor, nn
import os
import torch
import torch.nn as nn
# from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
import torch.nn.functional as F

class UMambaBlock(nn.Module):
    """
    UMambaBlock is a 5d Mamba block that can be used as a building block for a 5d visual model
    From the paper: https://arxiv.org/pdf/2401.04722.pdf

    Args:
        dim (int): The input dimension.
        dim_inner (Optional[int]): The inner dimension. If not provided, it is set to dim * expand.
        depth (int): The depth of the Mamba block.
        d_state (int): The state dimension. Default is 16.
        expand (int): The expansion factor. Default is 2.
        dt_rank (Union[int, str]): The rank of the temporal difference (Δ) tensor. Default is "auto".
        d_conv (int): The dimension of the convolutional kernel. Default is 4.
        conv_bias (bool): Whether to include bias in the convolutional layer. Default is True.
        bias (bool): Whether to include bias in the linear layers. Default is False.

    Examples::
        import torch
        # img:         B, C, H, W, D
        img_tensor = torch.randn(1, 64, 10, 10, 10)

        # Initialize Mamba block
        block = UMambaBlock(dim=64, depth=1)

        # Forward pass
        y = block(img_tensor)
        print(y.shape)

    """

    def __init__(
        self,
        dim: int = None,
        depth: int = 5,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.bias = bias

        # If dt_rank is not provided, set it to ceil(dim / d_state)
        dt_rank = math.ceil(self.dim / 16)
        self.dt_rank = dt_rank

        # If dim_inner is not provided, set it to dim * expand
        dim_inner = dim * expand
        self.dim_inner = dim_inner

        # If dim_inner is not provided, set it to dim * expand
        self.in_proj = nn.Linear(dim, dim_inner, bias=False)
        self.out_proj = nn.Linear(dim_inner, dim, bias=False)

        # Implement 2d convolutional layer
        # 3D depthwise convolution
        self.conv1 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim_inner,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=dim_inner,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        # Init instance normalization
        self.instance_norm = nn.InstanceNorm2d(dim)
        self.instance_norm2 = nn.InstanceNorm2d(dim_inner)

        # Leaky RELU
        self.leaky_relu = nn.LeakyReLU()

        # Layernorm
        self.norm = nn.LayerNorm(dim)

        # Mamba block
        self.mamba = MambaBlock(
            dim=dim,
            depth=depth,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            conv_bias=conv_bias,
            bias=bias,
        )

    def forward(self, x: Tensor):
        """
        B, C, H, W, D
        """
        b, c, h, w = x.shape
        input = x
        print(f"Input shape: {x.shape}")

        # Apply convolution
        x = self.conv1(x)
        print(f"Conv1 shape: {x.shape}")

        # # Instance Normalization
        x = self.instance_norm(x) + self.leaky_relu(x)
        print(f"Instance Norm shape: {x.shape}")

        # TODO: Add another residual connection here

        x = self.conv2(x)

        x = self.instance_norm(x) + self.leaky_relu(x)

        x = x + input

        # # Flatten to B, L, C
        x = rearrange(x, "b c h w -> b (h w) c")
        print(f"Faltten shape: {x.shape}")
        x = self.norm(x)

        # Maybe use a mamba block here then reshape back to B, C, H, W, D
        x = self.mamba(x)

        # Reshape back to B, C, H, W, D
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        return x
    


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
            UMambaBlock(out_channels, depth=5),
        )
        self.emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:,:, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels//2),
            # UMambaBlock(out_channels, depth=5),
        )

        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        # print('scale up', x.shape)
        x = torch.cat([skip_x, x], dim=1)

        x = self.conv(x)
        emb = self.emb(t)[:,:, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + emb
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels, size):
        super().__init__()
        self.channels = in_channels
        self.size = size
        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(in_channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm([in_channels]),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value

        return attention_value.swapaxes(1, 2).view(-1, self.channels, self.size, self.size)
    

class UNet_conditional_concat_with_mask(nn.Module):
    '''
    在输入上concat一次
    '''
    def __init__(self, c_in=3, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 48)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 24)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 12)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 24)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 48)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 96)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.mamba = UMambaBlock(dim=512,depth=5)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def forward(self, x, t, y, m):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print(y.shape)
        x = torch.concat([x, y, m], dim=1)
        # print(x.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # print('x2 shape', x2.shape)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        # print('x3 shape', x3.shape)
        x3 = self.sa2(x3)
        
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        # print('x4 shape', x4.shape)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.mamba(x4)
        x4 = self.bot3(x4)
        # print('x4 shape', x4.shape)

        x = self.up1(x4, x3, t)
        # print('x shape up 1', x.shape)
        x = self.sa4(x)
        # print('sa4 shape', x.shape)
        x = self.up2(x, x2, t)
        # print('x shape up 2', x.shape)
        x = self.sa5(x)
        # print('sa5 shape', x.shape)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)
        return output

# img_tensor = torch.randn(2, 1, 96, 96).cuda()

# # Initialize Mamba block
# unet = UNet_conditional_concat_with_mask().cuda()
# t = (torch.ones(2) * 3).long().cuda()
# Forward pass
# y = unet(img_tensor, t, img_tensor, img_tensor)