#from model.architectures import pertnas
import numpy as np
import tensorflow.keras.backend as K
#import torch

'''
a = pertnas.PertNAS()#.to(0)
a = a.to('cuda')
input = np.ones((1, 5, 128, 128))
input = torch.tensor(input).float().cuda()
#print(input)
l_outs, r_outs = a(input)
print('--=='*20)
for i in range(len(l_outs)):
    print(l_outs[i].size())

for i in range(len(r_outs)):
    print(r_outs[i].size())
'''

from model.architectures import ops_keras
import keras

a = ops_keras.OPS['sep_conv_5x5'](64, 1, affine=True, edge=1)
x = np.ones((1,128,128,64))
b = K.constant(x)
out = a(b)
