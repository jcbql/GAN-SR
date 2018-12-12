#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import OrderedDict
import torch
import torch.nn as nn

####################
# Basic blocks
####################


def act(act_type, neg_slope=0.2, inplace=True):
    # activation function
    # act_type: for selecting type: None -> no activation
    # neg_slope: for leakyrelu 
    if not act_type:
        return None
    if act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, act_type='leakyrelu'):
    """
    Conv layer with padding, normalization, activation
    mode: Conv -> Norm -> Act
    Not using normalization
    """

    padding = get_valid_padding(kernel_size, dilation)

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)

    a = act(act_type)

    p = None

    n = None 
    
    return sequential(p, c, n, a)




class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, act_type='leakyrelu'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, act_type=act_type)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, act_type=act_type)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, act_type=act_type)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, act_type=act_type)
        self.conv5 = conv_block(nc+4*gc, nc, 3, stride, bias=bias, act_type=None)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, act_type='leakyrelu'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, act_type)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, act_type)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, act_type)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


####################
# Upsampler
####################


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                 act_type='relu', mode='nearest'):
    # Up conv
    
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, act_type=act_type)
    return sequential(upsample, conv)


# In[ ]:




