from src.vit import *
m =  vit_2_4_32()
print(m)
from torchsummary import summary

summary(m.cuda(),(3,32,32))