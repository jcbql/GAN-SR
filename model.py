
# coding: utf-8

# In[2]:


import math
import torch
import torch.nn as nn
import block as B


class RRDB_Net(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, act_type='leakyrelu', res_scale=1):
        super(RRDB_Net, self).__init__()
        n_upscale = int(math.log(upscale, 2))

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, act_type=None)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, act_type=act_type) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, act_type=None)

        upsample_block = B.upconv_blcok

        upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)), *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x

