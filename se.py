
from __future__ import print_function

import math
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe
#model = L.Eltwise(model, conv1,operation = 'SUM')
def bn_relu_conv(bottom, ks, nout, stride, pad, dropout):
    batch_norm = L.BatchNorm(bottom, in_place=False, batch_norm_param=dict(use_global_stats=True))
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    conv = L.Convolution(relu, kernel_size=ks, stride=stride, 
                    num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    if dropout>0:
        conv = L.Dropout(conv, dropout_ratio=dropout)
    return conv


def se(bottom, planes):
    pooling = L.Pooling(bottom, pool=P.Pooling.AVE, global_pooling=True)
    fc1 = L.InnerProduct(pooling, num_output=planes, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    relu = L.ReLU(fc1, in_place=True)
    fc2 = L.InnerProduct(relu, num_output=planes, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    model = L.Sigmoid(fc2, in_place=True)
    se = L.Scale(bottom,model,scale_param=dict(axis=0))
    return se

def preactbottleneck(bottom, ks, nout, stride, pad, groups=1):
    batch_norm = L.BatchNorm(bottom, in_place=False, batch_norm_param=dict(use_global_stats=True))
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    conv = L.Convolution(relu, kernel_size=ks, stride=stride, group=groups,
                    num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    return conv

def shortcut(bottom, nout, stride):
    conv = L.Convolution(bottom, kernel_size=1, stride=stride, 
                    num_output=nout, pad=0, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
   
    return conv
  
def add_layer(bottom, in_filter,num_filter, stride):
    widen_factor = 4
    #print(bottom)
    if stride!=1 or in_filter != num_filter*widen_factor:
        x = shortcut(bottom, nout=num_filter*widen_factor, stride=stride)
    else:
        x = bottom
    
    conv = preactbottleneck(bottom, ks=1, nout=num_filter, stride=1, pad=0, groups=1)
    conv = preactbottleneck(conv, ks=3, nout=num_filter, stride=stride, pad=1, groups=16)
    conv = preactbottleneck(conv, ks=1, nout=num_filter*widen_factor, stride=1, pad=0, groups=16)
    conv = se(conv,num_filter*widen_factor)
    out = L.Eltwise(x, conv)
    return out

#change the line below to experiment with different setting
#depth -- must be 3n+4
#first_output -- channels before entering the first dense block, twice the growth_rate for DenseNet-BC
#growth_rate -- growth rate
#dropout -- set to 0 to disable dropout, non-zero number to set dropout rate
def resnetx(depth=[3,4,6,3], width=[32,64,128,256]):
    #data, label = L.Data(source=data_file, backend=P.Data.LMDB, batch_size=batch_size, ntop=2, 
    #          transform_param=dict(crop_size=32,mirror=True, mean_file="/home/bob/caffe-master/examples/cifar10/mean.binaryproto"))
    data, label = L.Data(source="/home/bob/caffe-master/examples/cifar10/cifar10_train_lmdb", backend=P.Data.LMDB, batch_size=64, ntop=2,
               transform_param=dict(crop_size=32,mirror=True, mean_file="/home/bob/caffe-master/examples/cifar10/mean.binaryproto"))   
    nchannels = 32
    model = L.Convolution(data, kernel_size=3, stride=1, num_output=nchannels,
                        pad=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    model = L.BatchNorm(model, in_place=False,batch_norm_param=dict(use_global_stats=True) )
    scale = L.Scale(model, bias_term=True, in_place=True)
    
    num_input=nchannels
    strides = [1] + [1]*(depth[0]-1)
    for stride in strides:        
        model = add_layer(model,num_input, width[0], stride)
        num_input = width[0]*4

    strides = [2] + [1]*(depth[1]-1)
    for stride in strides:        
        model = add_layer(model, num_input, width[1], stride)
        num_input = width[1]*4

    strides = [2] + [1]*(depth[2]-1)
    for stride in strides:        
        model = add_layer(model,num_input, width[2], stride)
        num_input = width[2]*4

    strides = [2] + [1]*(depth[3]-1)
    for stride in strides:        
        model = add_layer(model, num_input,width[3], stride)
        num_input = width[3]*4
        
    model = L.Pooling(model, pool=P.Pooling.AVE, global_pooling=True)
    model = L.InnerProduct(model, num_output=10, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    loss = L.SoftmaxWithLoss(model)
    #accuracy = L.Accuracy(model, label)
    return to_proto(loss)

def make_net():

    with open('resnetx-50-se.prototxt', 'w') as f:
        # change the path to your data. If it's not lmdb format, also change first line of densenet() function
        print(str(resnetx()), file=f)

    #with open('test-resnetx-se-50.prototxt', 'w') as f:
    #    print(str(resnetx('/home/bob/caffe-master/examples/cifar10/cifar10_test_lmdb', batch_size=50)), file=f)


if __name__ == '__main__':

    make_net()


