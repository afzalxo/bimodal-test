import keras
from keras import layers
from keras import Sequential
from keras.applications import keras_modules_injection
from keras.layers import Lambda
from model.architectures.ops_keras import *
#from model.architectures.utils import drop_path

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

DARTS_CIFAR10 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce= [('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

#        [('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 2), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))
'''
class Cell(layers.Layer):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction=False, reduction_prev=False, edges_per_node=[2,2,2,2], name=None):
        super(Cell, self).__init__()
        self._edges_per_node = edges_per_node
        self._cell_name = name
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, edge=0)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, padding='VALID', edge=0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, padding='VALID', edge=0)    
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)
        self._test0 = layers.Conv2D(filters=16, kernel_size=1, padding='VALID', data_format='channels_last', use_bias=False)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = []#nn.ModuleList()
        #print(op_names, indices)
        for name, index in zip(op_names, indices):
            print(name, C)
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True, edge=0)
            #self._ops += [op]
            self._ops.append(op)
        self._indices = indices

    def _compute(self, s0, s1):#, drop_prob=0.3):
        assert(self._steps == len(self._edges_per_node))
        print('Preprocessing...')
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        print('11'*20)
        print(s0, s1)
        #aao = aa(s1)
        states = [s0, s1]
        ops_list, state_list, out_list = [], [], []
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            print('==--1'*20)
            print(i, h1, h2, op2.C_in, op2.C_out, op2.kernel_size)
            h1 = op1(h1)
            print('==--2'*20)
            h2 = op2(h2)
            print('==--3'*20)
            #if self.training and drop_prob > 0.:
            #    if not isinstance(op1, Identity):
            #        h1 = drop_path(h1, drop_prob)
            #    if not isinstance(op2, Identity):
            #        h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
            
        return layers.Concatenate(axis=3, name=self._cell_name)([states[i] for i in self._concat]) 
'''
def cell(genotype, C_prev_prev, C_prev, C, s0, s1, reduction=False, reduction_prev=False, edges_per_node=[2,2,2,2], name=None):
    _cell_name = name
    if reduction_prev:
        preprocess0 = FactorizedReduce(C_prev_prev, C, edge=0)
    else:
        preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, padding='VALID', edge=0)
    preprocess1 = ReLUConvBN(C_prev, C, 1, 1, padding='VALID', edge=0)    
    if reduction:
        op_names, indices = zip(*genotype.reduce)
        concat = genotype.reduce_concat
    else:
        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat

    assert len(op_names) == len(indices)
    _steps = len(op_names) // 2
    _concat = concat
    multiplier = len(concat)

    _ops = []#nn.ModuleList()
    #print(op_names, indices)
    for name, index in zip(op_names, indices):
        print(name, C)
        stride = 2 if reduction and index < 2 else 1
        op = OPS[name](C, stride, True, edge=0)
        #self._ops += [op]
        _ops.append(op)
    _indices = indices

    def _compute(s0, s1):#, drop_prob=0.3):
        assert(_steps == len(edges_per_node))
        print('Preprocessing...')
        s0 = preprocess0(s0)
        s1 = preprocess1(s1)
        print('11'*20)
        print(s0, s1)
        states = [s0, s1]
        ops_list, state_list, out_list = [], [], []
        for i in range(self._steps):
            h1 = states[_indices[2*i]]
            h2 = states[_indices[2*i+1]]
            op1 = _ops[2*i]
            op2 = _ops[2*i+1]
            print('==--1'*20)
            print(i, h1, h2, op2.C_in, op2.C_out, op2.kernel_size)
            h1 = op1(h1)
            print('==--2'*20)
            h2 = op2(h2)
            print('==--3'*20)
            #if self.training and drop_prob > 0.:
            #    if not isinstance(op1, Identity):
            #        h1 = drop_path(h1, drop_prob)
            #    if not isinstance(op2, Identity):
            #        h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return states
    states = _compute(s0, s1)        
    return layers.Concatenate(axis=3, name=_cell_name)([states[i] for i in _concat]) 



'''
class PertNAS(layers.Layer):
    def __init__(self,
            classes=1000,
            cfg=None,
            genotype=DARTS_CIFAR10,
            **kwargs):
        super(PertNAS, self).__init__()

        #fusion_blocks = cfg.fusion_blocks    
        #x = Concatenate(axis=3, name='concat_0')([image_input, radar_input])
        C = 16
        C_curr = 48
        #self.stem = nn.Sequential(OrderedDict([('stem_conv',
        #    nn.Conv2d(5, C_curr, 3, padding=1, bias=False)),('stem_bn',
        #    nn.BatchNorm2d(C_curr))])
        #)
        self.stem = Sequential([layers.Conv2D(C_curr, kernel_size=3, padding='SAME', use_bias=False), layers.BatchNormalization(momentum=0.1, epsilon=1e-5)])
        multiplier = 4

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        # Block 1 - Image
        self.cells = []#nn.ModuleList()
        self.layer_outputs = []
        self.radar_outputs = []
        self.num_blocks = 5
        red_prev = False
        for i in range(self.num_blocks):
            self.cells.append(Cell(genotype, C_prev_prev, C_prev, C_curr, reduction=True, reduction_prev=red_prev, name='block'+str(i+1)+'_pool'))
            red_prev = True
            C_prev_prev, C_prev = C_prev, multiplier*C_curr+2
            if i is not self.num_blocks-2:
                C_curr *= 2


        #x = torch.cat((x, r), axis=3)

        #return x

    def call(self, input_tensor):
        #image_input = Lambda(lambda x: x[:, :, :, :3], name='image_channels')(input_tensor)
        #radar_input = Lambda(lambda x: x[:, :, :, 3:], name='radar_channels')(input_tensor)
        print(input_tensor)
        image_input = input_tensor#torch.tensor(input_tensor)
        #print(image_input.size())
        #exit(0)
        image_input = input_tensor[:,:,:,:3]
        radar_input = input_tensor[:,:,:,3:]
        #print(image_input.size())
        r = radar_input
        x = layers.Concatenate(axis=3, name='concat_0')([image_input, radar_input])
        x = x_prev = self.stem(x)
        #print(x.size(), x_prev.size())
        x = self.cells[0](x_prev, x)
        for i in range(1, self.num_blocks):
            r = layers.MaxPooling2D((2,2), strides=(2,2), name='rad_block'+str(i)+'_pool')(r)
            x = layers.Concatenate(axis=3, name='concat_'+str(i))([x,r])
            if i > 2:
                self.layer_outputs.append(x)
            self.radar_outputs.append(r)
            x_back = x
            x = self.cells[i](x_prev, x)
            x_prev = x_back 
        r = layers.MaxPooling2D((2,2), strides=(2,2), name='rad_block'+str(self.num_blocks)+'_pool')(r)
        x = layers.Concatenate(axis=3, name='concat_'+str(self.num_blocks)+'_pool')([x,r])
        self.layer_outputs.append(x)
        self.radar_outputs.append(r)
        #Radar outputs for blocks 6 and 7
        r = layers.MaxPooling2D((2,2), strides=(2,2), name='rad_block'+str(self.num_blocks+1)+'_pool')(r)
        self.radar_outputs.append(r)
        r = layers.MaxPooling2D((2,2), strides=(2,2), name='rad_block'+str(self.num_blocks+2)+'_pool')(r)
        self.radar_outputs.append(r)
        #print(x.size())

        return self.layer_outputs, self.radar_outputs
'''
@keras_modules_injection
def custom(*args, **kwargs):
    return pertnas(*args, **kwargs)

def pertnas(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, cfg=None, **kwargs):
    C = 16
    C_curr = 48
    #stem = Sequential([layers.Conv2D(C_curr, kernel_size=3, padding='SAME', use_bias=False), layers.BatchNormalization(momentum=0.1, epsilon=1e-5)])
    stem0 = layers.Conv2D(C_curr, kernel_size=3, padding='SAME', use_bias=False)
    stem1 = layers.BatchNormalization(momentum=0.1, epsilon=1e-5)
    multiplier = 4

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

    cells = []
    num_blocks = 5
    red_prev = False
    for i in range(num_blocks):
        print(C_prev_prev, C_prev, C)
        cells.append(Cell(DARTS_CIFAR10, C_prev_prev, C_prev, C_curr, reduction=True, reduction_prev=red_prev, name='block'+str(i+1)+'_pool'))
        red_prev = True
        C_prev_prev, C_prev = C_prev, multiplier*C_curr+2
        if i is not num_blocks-2:
            C_curr *= 2

    #input_tensor = layers.Input(tensor=input_tensor, shape=input_tensor.shape)
    image_input = Lambda(lambda x: x[:, :, :, :3], name='image_channels')(input_tensor)
    radar_input = Lambda(lambda x: x[:, :, :, 3:], name='radar_channels')(input_tensor)
    #print(image_input.size(), radar_input.size())
    layer_outputs = []
    radar_outputs = []

    r = radar_input
    x = layers.Concatenate(axis=3, name='concat_0')([image_input, radar_input])
    x_backup = x
    #print(x, input_tensor)
    #exit(0)
    x = x_prev = stem1(stem0(x))
    #x_prev = stem(x_backup)
    x = cells[0]._compute(x_prev, x)
    for i in range(1, num_blocks):
        r = layers.MaxPooling2D((2,2), strides=(2,2), name='rad_block'+str(i)+'_pool')(r)
        x = layers.Concatenate(axis=3, name='concat_'+str(i))([x,r])
        if i > 2:
            layer_outputs.append(x)
        radar_outputs.append(r)
        x_back = x
        x = cells[i]._compute(x_prev, x)
        x_prev = x_back 
    r = layers.MaxPooling2D((2,2), strides=(2,2), name='rad_block'+str(num_blocks)+'_pool')(r)
    x = layers.Concatenate(axis=3, name='concat_'+str(num_blocks)+'_pool')([x,r])
    layer_outputs.append(x)
    radar_outputs.append(r)
    #Radar outputs for blocks 6 and 7
    r = layers.MaxPooling2D((2,2), strides=(2,2), name='rad_block'+str(num_blocks+1)+'_pool')(r)
    radar_outputs.append(r)
    r = layers.MaxPooling2D((2,2), strides=(2,2), name='rad_block'+str(num_blocks+2)+'_pool')(r)
    radar_outputs.append(r)
    #print(x.size())
    model = models.Model(input_tensor, x, name='vgg16')

    return model#layer_outputs, radar_outputs

