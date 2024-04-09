import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
import torch.nn.functional as F

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
            DoubleConv(in_channels, out_channels, in_channels//2)
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


class CrossAttention(nn.Module):##需要做修改
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

    def forward(self, x, y):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        y = y.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        y_ln = self.ln(y)
        attention_value, _ = self.mha(x_ln, y_ln, y_ln)
        x = x + attention_value
        x = self.ff_self(x) + x
        
        return x.swapaxes(1, 2).view(-1, self.channels, self.size, self.size)
    
class ImageEncoder(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256):
        super(ImageEncoder, self).__init__()
        self.conv1 = DoubleConv(c_in, 64)
        self.conv2 = DoubleConv(64, c_out)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(c_out * 60 * 60, 1024)
        self.fc2 = nn.Linear(1024, time_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 1,1,240,240
        x = self.pool(x)# 1,1,120,120
        x = F.relu(self.conv2(x))
        x = self.pool(x)# 1,1,60,60
        x = x.view(-1, 60 *60)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # print('final emb shape',x.shape)
        return x

class UNet_conditional(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 120)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 60)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 30)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 60)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 120)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 240)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.image_encoder = ImageEncoder(c_in, c_out, self.time_dim)
        # if num_classes is not None:
        #     self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print('time embedding finished')
        if y is not None:
            t += self.image_encoder(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # print('x2 shape', x2.shape)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        # print('x3 shape', x3.shape)
        # x3 = self.sa2(x3)
        
        x4 = self.down3(x3, t)
        # x4 = self.sa3(x4)
        # print('x4 shape', x4.shape)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        # print('x4 shape', x4.shape)

        x = self.up1(x4, x3, t)
        # print('x shape up 1', x.shape)
        # x = self.sa4(x)
        # print('sa4 shape', x.shape)
        x = self.up2(x, x2, t)
        # print('x shape up 2', x.shape)
        # x = self.sa5(x)
        # print('sa5 shape', x.shape)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)
        return output