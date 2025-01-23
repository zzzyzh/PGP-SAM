
import torch
from torch import nn

from .common import get_norm, get_act


class ConvLayer2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3 or (3,3),
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        norm=None,
        act_func=None,
        inplace=False
    ):
        super(ConvLayer2d, self).__init__()

        padding = (kernel_size // 2) * dilation if isinstance(kernel_size, int) else tuple((x // 2) * dilation for x in kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.norm = get_norm(norm, out_channels) if norm != None else None
        self.act = get_act(act_func, inplace) if act_func != None else None
    
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, a=1)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm != None:
            x = self.norm(x)
        if self.act != None:
            x = self.act(x)
        
        return x


class ConvLayer1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_channels=None,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        norm=None,
        act_func=None
    ):
        super(ConvLayer1d, self).__init__()

        padding = (kernel_size // 2) * dilation if isinstance(kernel_size, int) else tuple((x // 2) * dilation for x in kernel_size)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.norm = get_norm(norm, norm_channels, False) if norm != None and norm_channels!= None else None
        self.act = get_act(act_func) if act_func != None else None
    
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, a=1)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm != None:
            x = self.norm(x)
        if self.act != None:
            x = self.act(x)
        
        return x


class UpSample2d(nn.Module):
    def __init__(self, 
                 in_planes: int,
                 out_planes: int,
                 norm_channels=None,
                 kernel_size=2,
                 stride=2,
                 bias=True,
                 norm=None,
                 act_func=None
                 ):
        super().__init__() 
        
        self.up_sample = nn.ConvTranspose2d(in_planes, 
                                            out_planes,
                                            kernel_size=kernel_size, 
                                            stride=stride,
                                            bias=bias
                                            )

        self.norm = get_norm(norm, norm_channels, False) if norm != None and norm_channels!= None else None
        self.act = get_act(act_func) if act_func != None else None
    
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.up_sample.weight, a=1)
        if self.up_sample.bias is not None:
            nn.init.constant_(self.up_sample.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_sample(x)
        if self.norm != None:
            x = self.norm(x)
        if self.act != None:
            x = self.act(x)
        
        return x

