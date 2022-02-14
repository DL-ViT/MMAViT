# from src.vit import *
# m =  vit_2_4_32()
# print(m)
# from torchsummary import summary
#
# summary(m.cuda(),(3,32,32))

#
# A = torch.rand(100, 3, 3).cuda()
# u, s, v = svd(A)
# u, s, v = torch.svd(A)
import numpy as np
import torch
import torch.nn as nn
from src import *
from src.grassmanian_vit import *
from src.img_gm_vit import *
from src.manifold_vit import *


# m = img_riem_vit_6_4_32()
# m = img_gm_vit_6_4_32()
# m = manifold_cct_7_3x1_32()
# m = manifold_vit_nano_12_p16()
# m = manifold_vit_small_12_p16()

m = manifold_vit_6_4_32(attention_type='riem')
m  = manifold_cct_7_3x2_32(attention_type='all')

#m = manifold_cvt_6_4_32(attention_type='all')
# m = nn.LayerNorm((8,64,64))

from pthflops import count_ops

inp = torch.rand(1,8,64,64)
inp = torch.rand(1,3,32,32)
print(m)
model_parameters = filter(lambda p: p.requires_grad, m.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
# Count the number of FLOPs
count_ops(m, inp)
# m = nn.LayerNorm((8, 196,196))
# model_parameters = filter(lambda p: p.requires_grad, m.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
print(f' Number of params :  {params}')
#print(m)

