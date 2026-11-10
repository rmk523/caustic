import os
import torch
import torch.nn as nn
import math
import numpy as np
from PIL import Image

w0 = 1
class SIRENActivation(nn.Module):
    def forward(self, x):
        return torch.sin(w0 * x)
class CausticNN(nn.Module):
    def __init__(self, input_dim, unfreeze_scalar, embedder_para, output_dim, layer_dims, skip_layers,
                  freeze_layers=[], scalar = 1.0, if_adafreq = True):
        super().__init__()
        self.embedder_dim = len(embedder_para) * 4
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()  # 使用 ModuleList 管理可学习参数
        self.skip = skip_layers
        layer_dims = [self.embedder_dim + 2] + layer_dims + [output_dim]  # 修正层维度
        self.if_adafreq = if_adafreq
        self.scalar = scalar
        self.unfreeze_scalar = None
        self.unfreeze_integer = None
        self.unfreeze_fraction = None
        self.freeze_dim = None
        self.separate_scalar(unfreeze_scalar)
        self.expand = 40.0/self.scalar

        self.part_freeze_layers = []

        # 初始化第一层
        linear = nn.Linear(input_dim, self.embedder_dim)
        self._init_first_layer(linear, embedder_para)
        self.layers.append(linear)
        self.activations.append(SIRENActivation())

        # 初始化其他层
        for i in range(1, len(layer_dims)):
            in_features = layer_dims[i-1]
            out_features = layer_dims[i]
            if i in skip_layers:
                out_features -= layer_dims[0]
            linear = nn.Linear(in_features, out_features)
            
            if i < len(layer_dims)-1:
                if i < len(layer_dims)-3:
                    if i == 1 or (i-1 in self.skip):
                        self.part_normal_init_layer(linear, gain=np.sqrt(6 / in_features)/w0, initial_length = (in_features-self.freeze_dim))
                        self.layers.append(linear)
                        self.activations.append(SIRENActivation())
                        self.part_freeze_layers.append(i)
                    else:
                        self.all_normal_init_layer(linear, gain=np.sqrt(6 / in_features)/w0)
                        self.layers.append(linear)
                        self.activations.append(SIRENActivation())
                else:
                    if i-1 in self.skip:
                        self.part_normal_init_layer(linear, gain=np.sqrt(2 / in_features)/50, initial_length = (in_features-self.freeze_dim))
                        self.layers.append(linear)
                        self.activations.append(nn.Sigmoid())
                        # self.activations.append(SIRENActivation())
                        self.part_freeze_layers.append(i)                      
                    else:
                        self.all_normal_init_layer(linear, gain=np.sqrt(2 / in_features))
                        self.layers.append(linear)
                        self.activations.append(nn.Sigmoid())
                        # self.activations.append(SIRENActivation())
            else:
                self.all_normal_init_layer(linear, gain=np.sqrt(2 / in_features)/50)
                self.layers.append(linear)

        # 冻结指定层
        for idx in freeze_layers:
            for param in self.layers[idx].parameters():
                param.requires_grad = False


    def separate_scalar(self, unfreeze_scalar):
        self.unfreeze_scalar = unfreeze_scalar
        scalar_value = float(self.unfreeze_scalar)
        rounded_value = int(round(scalar_value))
        self.unfreeze_integer = rounded_value
        self.unfreeze_fraction = 0.0
        self.freeze_dim = self.embedder_dim - 4 * self.unfreeze_integer

    def zero_initial_layer(self):
        for layer_idx in self.part_freeze_layers:
            layer = self.layers[layer_idx]
            start_col = layer.weight.shape[1] - self.freeze_dim
            end_col = layer.weight.shape[1]
            layer.weight.data[:, start_col:end_col] = 0

    def _init_first_layer(self, linear, embedder_para):
        rows = []
        for val in embedder_para:
            # scaled_val = val / 40 * math.pi/w0
            scaled_val = val * math.pi/w0/(self.scalar)
            rows.extend([[scaled_val, 0.0]] * 2)
            rows.extend([[0.0, scaled_val]] * 2)
        spiral_values = torch.tensor(rows, dtype=torch.float32)
        linear.weight.data = spiral_values

        if linear.bias is not None:
            num_biases = linear.bias.shape[0]
            even_indices = torch.arange(1, num_biases+1, 2)
            linear.bias.data.zero_()
            linear.bias.data[even_indices] = math.pi / 2

    def all_uniform_init_layer(self, linear, b):
        nn.init.uniform_(linear.weight, -b, b)
        nn.init.uniform_(linear.bias, -math.pi, math.pi) 
    
    def part_uniform_init_layer(self, linear, b, initial_length):
        torch.nn.init.constant_(linear.weight[:, initial_length:], 0.0)
        nn.init.uniform_(linear.weight[:, :initial_length], -b, b)
        nn.init.uniform_(linear.bias, -math.pi, math.pi) 

    def all_normal_init_layer(self, linear, gain):
        nn.init.normal_(linear.weight, 0.0, gain)
        nn.init.zeros_(linear.bias)
    
    def part_normal_init_layer(self, linear, gain, initial_length, gain2 = 0):
        nn.init.normal_(linear.weight[:, initial_length:], 0.0, gain2)
        nn.init.normal_(linear.weight[:, :initial_length], 0.0, gain)
        nn.init.zeros_(linear.bias)

    def zero_init_layer(self, linear):
        # torch.nn.init.constant_(linear.weight[:, initial_length:], 0.0)
        nn.init.constant_(linear.weight, 0.0)
        nn.init.constant_(linear.bias, 1.0)

    def forward(self, x):
        x_embedder = self.layers[0](x)
        x_embedder = self.activations[0](x_embedder)

        total_columns = x_embedder.size(-1)
        mask = torch.ones_like(x_embedder)
        start_idx = 4 * self.unfreeze_integer
        mask[..., start_idx:] = 0
        # end_idx = start_idx + 4
        # if start_idx < total_columns:
        #     actual_end = min(end_idx, total_columns)
        #     mask[..., start_idx:actual_end] = self.unfreeze_fraction
        # if end_idx < total_columns:
        #     mask[..., end_idx:] = 0
        x_embedder = x_embedder * mask

        x_embedder = torch.cat([x, x_embedder], 1) # 保存嵌入层输出
        # x_embedder = torch.cat([x*self.expand, x_embedder], 1)
        
        x = x_embedder
        for layer_idx in range(1, len(self.layers)):
            layer = self.layers[layer_idx]
            activation = self.activations[layer_idx] if layer_idx < len(self.activations) else None

            # 处理跳跃连接
            if layer_idx-1 in self.skip:
                x = torch.cat([x, x_embedder/ np.sqrt(layer_idx-1)], dim=-1)
            
            x = layer(x)
            if activation is not None:
                x = activation(x)
        return x

    def updata_w0(self, w0):
        self.w0 = w0