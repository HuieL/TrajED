import math
import os
import urllib.request
from functools import partial
from urllib.error import HTTPError

import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm.notebook import tqdm
import ipdb


# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class fixed_embed(nn.Module):
    def __init__(self, dim_feedforward, freq, num_kernels, input_dim_st, input_dim_text, alpha, scaler):
        """Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
        """
        super().__init__()
        self.freq = freq
        self.alpha = alpha
        self.input_dim_st = input_dim_st
        self.input_dim_text = input_dim_text
        self.scaler = scaler

        self.conv_st = nn.Sequential(
            Inception_Block(input_dim_st, dim_feedforward,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block(dim_feedforward, input_dim_st,
                               num_kernels=num_kernels),
        )

        self.conv_text = nn.Sequential(
            Inception_Block(input_dim_text, dim_feedforward,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block(dim_feedforward, input_dim_text,
                               num_kernels=num_kernels),
        )

    def conv_embed(self, x):
        B, T, N = x.size()
        if N == self.input_dim_st: conv =  self.conv_st
        elif N == self.input_dim_text: conv = self.conv_text
        else:
            raise ValueError
        # padding
        if T % self.freq != 0:
            length = ((T // self.freq) + 1) * self.freq
            padding = torch.zeros([x.shape[0], (length - T), x.shape[2]]).to(x.device)
            padding_x = torch.cat([x, padding], dim=1)
        else:
            length = T
            padding_x = x
        # reshape
        out = padding_x.reshape(B, length // self.freq, self.freq, N).permute(0, 3, 1, 2).contiguous()
        # 2D conv: from 1d Variation to 2d Variation: [B, N, T/freq, freq]
        out = conv(out)
        # reshape to feq: [B, freq, N]
        out = torch.mean(out.permute(0, 2, 3, 1), dim=1)
        # residual connection
        conv_embeddings = out + torch.mean(padding_x.reshape(B, length // self.freq, self.freq, N).contiguous(), dim=1)
        return conv_embeddings
    
    def fix(self, st_embeddings, text_embeddings):
        fix_embeddings = torch.cat(((1-self.alpha) * self.scaler * st_embeddings, self.alpha * text_embeddings), dim = 2)
        return fix_embeddings

    def forward(self, x, t):
        # st_embeddings:
        st_embeddings = self.conv_embed(x)
        # text_embeddings:
        text_embeddings = self.conv_embed(t)
        #fix_embeddings:
        fix_embeddings = self.fix(st_embeddings, text_embeddings)
        return fix_embeddings
    
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class BiraryAttn(nn.Module):
    def __init__(self, threhold = 0):
        super(BiraryAttn, self).__init__()
        self.threhold = threhold

    def forward(self, x):
        return (x>self.threhold).float()

class Trajformer(nn.Module):
    def __init__(self, num_classes = 2, num_layers = 3, dim_hidden = 16, freq = 30, num_kernels = 6, input_dim_st = 4, input_dim_text = 768, alpha = 0.0, scaler = 10e-7, num_heads = 1, dropout=0.0, attn_threhold = 0, return_attention = True):
        super().__init__()
        self.num_heads = num_heads
        self.return_attention = return_attention
        self.encoder_input_dim = input_dim_st + input_dim_text

        self.emded = fixed_embed(dim_feedforward = dim_hidden, input_dim_st = input_dim_st, input_dim_text = input_dim_text, freq = freq, num_kernels = num_kernels, alpha = alpha, scaler = scaler)
        self.layers = nn.ModuleList([EncoderBlock(input_dim = self.encoder_input_dim, num_heads = num_heads, dim_feedforward = dim_hidden, dropout = dropout) for _ in range(num_layers)])
        self.attn_linear = nn.Sequential(nn.Linear(freq, 1, bias=True),
                           nn.Sigmoid())
        self.binary_attn = BiraryAttn(attn_threhold)
        self.projection = nn.Linear(self.encoder_input_dim * freq, num_classes)

    def forward(self, x, t, mask=None):
        attention_maps = []
        #Get Fixed embeddings:
        x_ = self.emded(x, t)
        #Padding for multi_head attention:
        if x_.shape[-1] % self.num_heads != 0:
            length = ((x_.shape[-1] // self.num_heads) + 1) * self.num_heads
            padding = torch.zeros([x_.shape[0], (length - x_.shape[-1]), x_.shape[2]])
            x_ = torch.cat([x_, padding], dim=1)
        #Encoder layer:
        for layer in self.layers:
            x_ = layer(x_, mask=mask)
            if self.return_attention:
                _, attn_map = layer.self_attn(x_, mask=mask, return_attention=self.return_attention)
                attention_maps.append(attn_map)
        attn = self.binary_attn(self.attn_linear(attention_maps[-1].squeeze(1)).squeeze(-1))
        #Projection layer:
        x_ = self.projection(x_.reshape(x_.shape[0], -1))
        if self.return_attention: 
            return x_, attn
        else: 
            return x_
