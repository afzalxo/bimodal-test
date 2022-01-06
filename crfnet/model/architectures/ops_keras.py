#from tensorflow.keras import Sequential
from collections import OrderedDict
import tensorflow as tf
import keras
from keras import layers
from keras import Sequential
from keras import backend as K

OPS = {
  #'none' : lambda C, stride, affine, edge: Zero(stride),
  'skip_connect' : lambda C, stride, affine, edge: Identity() if stride == 1 else FactorizedReduce(C, C, edge, affine=affine),
  'max_pool_3x3' : lambda C, stride, affine, edge: layers.MaxPooling2D((3, 3), strides=stride, padding='SAME'),
  'avg_pool_3x3' : lambda C, stride, affine, edge: layers.AveragePooling2D((3, 3), strides=stride, padding='SAME'),
  'sep_conv_3x3' : lambda C, stride, affine, edge: SepConv(C, C, 3, stride, 'SAME', edge, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine, edge: SepConv(C, C, 5, stride, 'SAME', edge, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine, edge: DilConv(C, C, 3, stride, 'SAME', 2, edge, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine, edge: DilConv(C, C, 5, stride, 'SAME', 2, edge, affine=affine)
}
class ReLUConvBN(layers.Layer):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, edge):
    super(ReLUConvBN, self).__init__()
    self.rel = layers.ReLU()
    self.con1 = layers.Conv2D(C_out, kernel_size, strides=(stride, stride), padding=padding, use_bias=False)
    self.bn1 = layers.BatchNormalization(momentum=0.1, epsilon=1e-5)
    '''
    self.op = Sequential(
             [layers.ReLU(), 
              layers.Conv2D(C_out, kernel_size, strides=(stride, stride), padding=padding, use_bias=False), 
              layers.BatchNormalization(momentum=0.1, epsilon=1e-5)])
    '''
  def call(self, x):
    x = self.rel(x)
    x = self.con1(x)
    x = self.bn1(x)
    return x
    #return self.op(x)

class DilConv(layers.Layer):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, edge, affine=True):
    super(DilConv, self).__init__()
    
    self.op = Sequential(
            [layers.ReLU(), 
             #layers.Conv2D(C_in, kernel_size=kernel_size, strides=stride, padding=padding, dilation_rate=dilation, groups=C_in, use_bias=False), 
             layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding, dilation_rate=dilation, use_bias=False), 
             layers.Conv2D(C_out, kernel_size=1, padding='VALID', use_bias=False), 
             layers.BatchNormalization(momentum=0.1, epsilon=1e-5)]
            )

  def call(self, x):
    return self.op(x)


class SepConv(layers.Layer):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, edge, affine=True):
    super(SepConv, self).__init__()
    self.C_in = C_in
    self.C_out = C_out
    self.kernel_size = kernel_size
    self.relu1 = layers.ReLU()
    self.dc1 = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding, data_format='channels_last', use_bias=False) 
    self.conv2d1 = layers.Conv2D(filters=C_in, kernel_size=1, padding='VALID', data_format='channels_last', use_bias=False)
    self.bn1 = layers.BatchNormalization(momentum=0.1, epsilon=1e-5)
    self.relu2 = layers.ReLU()
    print('---'*20)
    print(self.dc1, self.C_in, self.C_out, self.kernel_size)
    self.dc2 = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=1, padding=padding, data_format='channels_last', use_bias=False) 
    self.conv2d2 = layers.Conv2D(filters=C_out, kernel_size=1, padding='VALID', data_format='channels_last', use_bias=False)
    self.bn2 = layers.BatchNormalization(momentum=0.1, epsilon=1e-5)
    '''
    self.op = Sequential(
            [layers.ReLU(), 
             layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding=padding, data_format='channels_last', use_bias=False), 
             layers.Conv2D(C_in, kernel_size=1, padding='VALID', use_bias=False),
             layers.BatchNormalization(momentum=0.1, epsilon=1e-5),
             layers.ReLU(), 
             layers.DepthwiseConv2D(kernel_size=kernel_size, strides=1, padding=padding, data_format='channels_last', use_bias=False), 
             layers.Conv2D(C_out, kernel_size=1, padding='VALID', use_bias=False), 
             layers.BatchNormalization(momentum=0.1, epsilon=1e-5)]
            )
    '''

  def call(self, x):
    x = self.relu1(x)
    print('---'*10)
    print(x, self.dc1, self.dc1.kernel_size, self.dc1.filters)
    x = self.dc1(x)
    print('<<<'*10)
    x = self.conv2d1(x)
    print('>>>'*10)
    x = self.bn1(x)
    x = self.relu2(x)
    x = self.dc2(x)
    x = self.conv2d2(x)
    x = self.bn2(x)
    return x

    #return self.op(x)


class Identity(layers.Layer):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

'''
class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)
'''

class FactorizedReduce(layers.Layer):

  def __init__(self, C_in, C_out, edge, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = Sequential([layers.ReLU()])
    self.conv_1 = Sequential([layers.Conv2D(C_out // 2, 1, strides=2, padding='VALID', use_bias=False)])
    self.conv_2 = Sequential([layers.Conv2D(C_out // 2, 1, strides=2, padding='VALID', use_bias=False)])
    self.bn = Sequential([layers.BatchNormalization(momentum=0.1, epsilon=1e-5)]) 

  def call(self, x):
    x = self.relu(x)
    out = layers.Concatenate(axis=3)([self.conv_1(x), self.conv_2(x[:,1:,1:,:])])
    out = self.bn(out)
    return out

