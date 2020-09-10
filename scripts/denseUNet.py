# This program generates denseUNet's train, test, solver prototxt 

# +---+----+----+----+----+----+----+----+
# | 0 | 8  | 16 | 24 | 32 | 40 | 48 | 56 |
# +---+----+----+----+----+----+----+----+
# | 1 | 9  | 17 | 25 | 33 | 41 | 49 | 57 |
# +---+----+----+----+----+----+----+----+
# | 2 | 10 | 18 | 26 | 34 | 42 | 50 | 58 |
# +---+----+----+----+----+----+----+----+
# | 3 | 11 | 19 | 27 | 35 | 43 | 51 | 59 |
# +---+----+----+----+----+----+----+----+
# | 4 | 12 | 20 | 28 | 36 | 44 | 52 | 60 |
# +---+----+----+----+----+----+----+----+
# | 5 | 13 | 21 | 29 | 37 | 45 | 53 | 61 |
# +---+----+----+----+----+----+----+----+
# | 6 | 14 | 22 | 30 | 38 | 46 | 54 | 62 |
# +---+----+----+----+----+----+----+----+
# | 7 | 15 | 23 | 31 | 39 | 47 | 55 | 63 |
# +---+----+----+----+----+----+----+----+

# +---+----+----+----+----+
# | 0 |  5 | 10 | 15 | 20 |
# +---+----+----+----+----+
# | 1 |  6 | 11 | 16 | 21 |
# +---+----+----+----+----+
# | 2 |  7 | 12 | 17 | 22 |
# +---+----+----+----+----+
# | 3 |  8 | 13 | 18 | 23 |
# +---+----+----+----+----+
# | 4 |  9 | 14 | 19 | 24 |
# +---+----+----+----+----+

import os
import sys
import cv2
import numpy as np
import json
import caffe
from caffe.io import caffe_pb2
from caffe import layers as L
from caffe import params as P

np.set_printoptions(threshold=sys.maxsize)
caffe.set_mode_gpu()

CENTER_SAI = 12
SHIFT_VALUE = 0.8

def shift_value_5x5(i):
    if i<=4:
        tx = -2*SHIFT_VALUE
    elif i>4 and i<=9:
        tx = -1*SHIFT_VALUE
    elif i>9 and i<=14:
        tx = 0
    elif i>14 and i<=19:
        tx = 1*SHIFT_VALUE
    elif i>19 and i<=24:
        tx = 2*SHIFT_VALUE
    else:
        tx = 3*SHIFT_VALUE
    if i == 0 or (i)%5==0:
        ty = -2*SHIFT_VALUE
    elif i == 1 or (i-1)%5==0:
        ty = -1*SHIFT_VALUE
    elif i == 2 or (i-2)%5==0:
        ty = 0
    elif i == 3 or (i-3)%5==0:
        ty = 1*SHIFT_VALUE
    elif i == 4 or (i-4)%5==0:
        ty = 2*SHIFT_VALUE
    else:
        ty = 3*SHIFT_VALUE
    return tx, ty

def shift_value_8x8(i, shift_val):
    if i<=7:
        tx = -3*shift_val
    elif i>7 and i<=15:
        tx = -2*shift_val
    elif i>15 and i<=23:
        tx = -1*shift_val
    elif i>23 and i<=31:
        tx = 0
    elif i>31 and i<=39:
        tx = 1*shift_val
    elif i>39 and i<=47:
        tx = 2*shift_val
    elif i>47 and i<=55:
        tx = 3*shift_val
    else:
        tx = 4*shift_val
    if i==0 or (i%8==0 and i>7):
        ty = -3*shift_val
    elif i == 1 or (i-1)%8==0:
        ty = -2*shift_val
    elif i == 2 or (i-2)%8==0:
        ty = -1*shift_val
    elif i == 3 or (i-3)%8==0:
        ty = 0
    elif i == 4 or (i-4)%8==0:
        ty = 1*shift_val
    elif i == 5 or (i-5)%8==0:
        ty = 2*shift_val
    elif i == 6 or (i-6)%8==0:
        ty = 3*shift_val
    else:
        ty = 4*shift_val
    return tx, ty
    
def index_5x5_picker(i):
    list_5x5 = [9, 10, 11, 12, 13,
                17, 18, 19, 20, 21,
                25, 26, 27, 28, 29,
                33, 34, 35, 36, 37,
                41, 42, 43, 44, 45]
    i_dst = list_5x5[i]
    return i_dst

def flow_layer(bottom=None, nout=1):
    conv = L.Convolution(bottom, kernel_size=3, stride=1,
                                num_output=nout, pad=1, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True)
    conv = L.Convolution(conv, kernel_size=3, stride=1,
                                num_output=nout, pad=1, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True)
    conv = L.Convolution(conv, kernel_size=1, stride=1,
                                num_output=nout, pad=0, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    return conv

def conv_layer(bottom=None, ks=3, nout=1, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True)
    return conv

def conv_final_layer(bottom=None, ks=3, nout=1, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.TanH(conv, in_place=True)
    return conv

def conv_conv_layer(bottom=None, ks=3, nout=1, stride=1, pad=1, drop=0.5):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True)
    if drop != 0: 
        conv = L.Dropout(conv, dropout_param=dict(dropout_ratio=drop))
    conv = L.Convolution(conv, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True)
    return conv

def downsample_layer(bottom=None, ks=3, stride=2):
    pool = L.Pooling(bottom, kernel_size=ks, stride=stride, pool=P.Pooling.MAX)
    pool = L.ReLU(pool, relu_param=dict(negative_slope=0.2), in_place=True)
    return pool

def upsample_layer(bottom=None, ks=2, nout=1, stride=2):
    deconv = L.Deconvolution(bottom, convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride, pad=0))
    deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=True)
    return deconv

def upsample_concat_layer(bottom1=None, bottom2=None, ks=2, nout=1, stride=2, pad=0, batch_size=1, crop_size=0):
    deconv = L.Deconvolution(bottom1, convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride, pad=pad))
    deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=True)
    dum = L.DummyData(shape=dict(dim=[batch_size, nout, crop_size, crop_size]))
    deconv_crop = L.Crop(deconv, dum, crop_param=dict(axis=2, offset=0))
    conc = L.Concat(*[deconv_crop, bottom2], concat_param={'axis': 1})
    return conc

def upsample_concat_crop_layer(bottom1=None, bottom2=None, ks=2, nout=1, stride=2, pad=0, batch_size=1, crop_size=0):
    deconv = L.Deconvolution(bottom1, convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride, pad=pad))
    deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=True)
    dum = L.DummyData(shape=dict(dim=[batch_size, 64, 192, 192]))
    deconv_crop = L.Crop(deconv, dum, crop_param=dict(axis=2, offset=0))
    conc = L.Concat(*[deconv_crop, bottom2], concat_param={'axis': 1})
    return conc

def upsample_conv_conv_layer(bottom=None, ks=3, nout=1, stride=2, pad=1):
    deconv = L.Deconvolution(bottom, convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride))
    deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=True)
    conv = L.Convolution(deconv, kernel_size=ks, stride=1,
                                num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True)
    conv = L.Convolution(conv, kernel_size=ks, stride=1,
                                num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.TanH(conv, in_place=False)
    return conv

def conv_conv_downsample_layer(bottom=None, ks=3, nout=1, stride=2, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=1,
                                num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True)
    conv = L.Convolution(conv, kernel_size=ks, stride=1,
                                num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True)
    pool = L.Pooling(conv, kernel_size=ks, stride=stride, pool=P.Pooling.MAX)
    pool = L.ReLU(pool, relu_param=dict(negative_slope=0.2), in_place=True)
    return conv, pool

def conv_relu(bottom, ks, nout, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    relu = L.ReLU(conv, relu_param=dict(negative_slope=0.0), in_place=True)
    return relu

def block(bottom, ks, nout, dilation=1, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, dilation=dilation, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    relu = L.ReLU(conv, relu_param=dict(negative_slope=0.1), in_place=True)
    ccat = L.Concat(relu, bottom, axis=1)
    relu = L.ReLU(ccat, relu_param=dict(negative_slope=0.0), in_place=True)
    return relu

def fc_block(bottom, nout):
    fc = L.InnerProduct(bottom,
                        num_output=nout,
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type="gaussian", std=0.005),
                        bias_filler=dict(type="constant", value=1))
    relu = L.ReLU(fc, in_place=True)
    drop = L.Dropout(relu, dropout_ratio=0.5, in_place=True)
    return drop

def input_shifting_5x5(bottom, batch_size):
    con = None
    for i in range(25):
        center = bottom
        tx, ty = shift_value_5x5(i)
        param_str = json.dumps({'tx': tx, 'ty': ty})
        shift = L.Python(center, module = 'input_shifting_layer', layer = 'InputShiftingLayer', ntop = 1, param_str = param_str)
        if i == 0:
            con = shift
        else:   
            con = L.Concat(*[con, shift], concat_param={'axis': 1})
    dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    con_crop = L.Crop(con, dum_input, crop_param=dict(axis=2, offset=2))
    return con_crop

def image_data_5x5(batch_size=1, source='train_source'):
    con = None
    for i in range(25):
        label, trash = L.ImageData(batch_size=batch_size,
                                source='./datas/flower_dataset/'+source+str(i)+'.txt',
                                transform_param=dict(scale=1./256.),
                                shuffle=False,
                                ntop=2,
                                new_height=196,
                                new_width=196,
                                is_color=False)
        if i == 0:
            con = label
        else:   
            con = L.Concat(*[con, label], concat_param={'axis': 1})
    dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    con_crop = L.Crop(con, dum_input, crop_param=dict(axis=2, offset=2))
    return con_crop, trash

def ver_mean_block(bottom):
    ver_remain = bottom

    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean0 = ver_slice
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean0 = L.Eltwise(ver_mean0, ver_slice, operation=P.Eltwise.SUM)
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean0 = L.Eltwise(ver_mean0, ver_slice, operation=P.Eltwise.SUM)
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean0 = L.Eltwise(ver_mean0, ver_slice, operation=P.Eltwise.SUM)
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean0 = L.Eltwise(ver_mean0, ver_slice, operation=P.Eltwise.SUM)
    ver_mean0 = L.Power(ver_mean0, power=1.0, scale=1., shift=0.0, in_place=False)

    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean1 = ver_slice
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean1 = L.Eltwise(ver_mean1, ver_slice, operation=P.Eltwise.SUM)
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean1 = L.Eltwise(ver_mean1, ver_slice, operation=P.Eltwise.SUM)
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean1 = L.Eltwise(ver_mean1, ver_slice, operation=P.Eltwise.SUM)
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean1 = L.Eltwise(ver_mean1, ver_slice, operation=P.Eltwise.SUM)
    ver_mean1 = L.Power(ver_mean1, power=1.0, scale=1., shift=0.0, in_place=False)

    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean2 = ver_slice
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean2 = L.Eltwise(ver_mean2, ver_slice, operation=P.Eltwise.SUM)
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean2 = L.Eltwise(ver_mean2, ver_slice, operation=P.Eltwise.SUM)
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean2 = L.Eltwise(ver_mean2, ver_slice, operation=P.Eltwise.SUM)
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean2 = L.Eltwise(ver_mean2, ver_slice, operation=P.Eltwise.SUM)
    ver_mean2 = L.Power(ver_mean2, power=1.0, scale=1., shift=0.0, in_place=False)

    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean3 = ver_slice
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean3 = L.Eltwise(ver_mean3, ver_slice, operation=P.Eltwise.SUM)
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean3 = L.Eltwise(ver_mean3, ver_slice, operation=P.Eltwise.SUM)
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean3 = L.Eltwise(ver_mean3, ver_slice, operation=P.Eltwise.SUM)
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean3 = L.Eltwise(ver_mean3, ver_slice, operation=P.Eltwise.SUM)
    ver_mean3 = L.Power(ver_mean3, power=1.0, scale=1., shift=0.0, in_place=False)

    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean4 = ver_slice
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean4 = L.Eltwise(ver_mean4, ver_slice, operation=P.Eltwise.SUM)
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean4 = L.Eltwise(ver_mean4, ver_slice, operation=P.Eltwise.SUM)
    ver_slice, ver_remain = L.Slice(ver_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    ver_mean4 = L.Eltwise(ver_mean4, ver_slice, operation=P.Eltwise.SUM)
    ver_slice = ver_remain
    ver_mean4 = L.Eltwise(ver_mean4, ver_slice, operation=P.Eltwise.SUM)
    ver_mean4 = L.Power(ver_mean4, power=1.0, scale=1., shift=0.0, in_place=False)

    con = ver_mean0
    con = L.Concat(*[con, ver_mean1], concat_param={'axis': 1})
    con = L.Concat(*[con, ver_mean2], concat_param={'axis': 1})
    con = L.Concat(*[con, ver_mean3], concat_param={'axis': 1})
    con = L.Concat(*[con, ver_mean4], concat_param={'axis': 1})
    return con

def hor_mean_block(bottom):
    hor_remain = bottom

    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean0 = hor_slice
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean1 = hor_slice
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean2 = hor_slice
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean3 = hor_slice
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean4 = hor_slice

    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean0 = L.Eltwise(hor_mean0, hor_slice, operation=P.Eltwise.SUM)
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean1 = L.Eltwise(hor_mean1, hor_slice, operation=P.Eltwise.SUM)
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean2 = L.Eltwise(hor_mean2, hor_slice, operation=P.Eltwise.SUM)
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean3 = L.Eltwise(hor_mean3, hor_slice, operation=P.Eltwise.SUM)
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean4 = L.Eltwise(hor_mean4, hor_slice, operation=P.Eltwise.SUM)

    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean0 = L.Eltwise(hor_mean0, hor_slice, operation=P.Eltwise.SUM)
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean1 = L.Eltwise(hor_mean1, hor_slice, operation=P.Eltwise.SUM)
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean2 = L.Eltwise(hor_mean2, hor_slice, operation=P.Eltwise.SUM)
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean3 = L.Eltwise(hor_mean3, hor_slice, operation=P.Eltwise.SUM)
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean4 = L.Eltwise(hor_mean4, hor_slice, operation=P.Eltwise.SUM)

    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean0 = L.Eltwise(hor_mean0, hor_slice, operation=P.Eltwise.SUM)
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean1 = L.Eltwise(hor_mean1, hor_slice, operation=P.Eltwise.SUM)
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean2 = L.Eltwise(hor_mean2, hor_slice, operation=P.Eltwise.SUM)
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean3 = L.Eltwise(hor_mean3, hor_slice, operation=P.Eltwise.SUM)
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean4 = L.Eltwise(hor_mean4, hor_slice, operation=P.Eltwise.SUM)

    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean0 = L.Eltwise(hor_mean0, hor_slice, operation=P.Eltwise.SUM)
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean1 = L.Eltwise(hor_mean1, hor_slice, operation=P.Eltwise.SUM)
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean2 = L.Eltwise(hor_mean2, hor_slice, operation=P.Eltwise.SUM)
    hor_slice, hor_remain = L.Slice(hor_remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
    hor_mean3 = L.Eltwise(hor_mean3, hor_slice, operation=P.Eltwise.SUM)
    hor_slice = hor_remain
    hor_mean4 = L.Eltwise(hor_mean4, hor_slice, operation=P.Eltwise.SUM)

    hor_mean0 = L.Power(hor_mean0, power=1.0, scale=1., shift=0.0, in_place=False)
    hor_mean1 = L.Power(hor_mean1, power=1.0, scale=1., shift=0.0, in_place=False)
    hor_mean2 = L.Power(hor_mean2, power=1.0, scale=1., shift=0.0, in_place=False)
    hor_mean3 = L.Power(hor_mean3, power=1.0, scale=1., shift=0.0, in_place=False)
    hor_mean4 = L.Power(hor_mean4, power=1.0, scale=1., shift=0.0, in_place=False)

    con = hor_mean0
    con = L.Concat(*[con, hor_mean1], concat_param={'axis': 1})
    con = L.Concat(*[con, hor_mean2], concat_param={'axis': 1})
    con = L.Concat(*[con, hor_mean3], concat_param={'axis': 1})
    con = L.Concat(*[con, hor_mean4], concat_param={'axis': 1})
    return con
    
def slice_warp(shift, flow_h, flow_v):
    for i in range(25):
        if i < 24:
            shift_slice, shift = L.Slice(shift, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
            flow_h_slice, flow_h = L.Slice(flow_h, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
            flow_v_slice, flow_v = L.Slice(flow_v, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
            predict_slice = L.Warping(shift_slice, flow_h_slice, flow_v_slice)
        else:
            shift_slice = shift
            flow_h_slice = flow_h
            flow_v_slice = flow_v
            predict_slice = L.Warping(shift_slice, flow_h_slice, flow_v_slice)

        if i == 0:
            con = predict_slice
        else:
            con = L.Concat(*[con, predict_slice], concat_param={'axis': 1})
    return con

def slicer(bottom, dim):
    for i in range(192):
        if i == 0:
            bottom_slice, remain = L.Slice(bottom, ntop=2, slice_param=dict(slice_dim=dim, slice_point=[1]))
            con = bottom_slice
        elif i < 191:
            bottom_slice, remain = L.Slice(remain, ntop=2, slice_param=dict(slice_dim=dim, slice_point=[1]))
            con = L.Concat(*[con, bottom_slice], concat_param={'axis': 1})
        else:
            con = L.Concat(*[con, remain], concat_param={'axis': 1})

    return con

def luma_layer(bottom):
    r, g, b = L.Slice(bottom, ntop=3, slice_param=dict(slice_dim=2, slice_point=[1,2]))
    y_r = L.Power(r, power=1.0, scale=0.299, shift=0.0, in_place=False)
    y_g = L.Power(g, power=1.0, scale=0.587, shift=0.0, in_place=False)
    y_b = L.Power(b, power=1.0, scale=0.114, shift=0.0, in_place=False)
    y = L.Eltwise(y_r, y_g, operation=P.Eltwise.SUM)
    y = L.Eltwise(y, y_b, operation=P.Eltwise.SUM)        
    return y

def denseUNet_train(batch_size=1):
    n = caffe.NetSpec()

    # Data loading
    n.input, n.trash = L.ImageData(batch_size=batch_size,
                            source='./datas/flower_dataset/train_source'+str(CENTER_SAI)+'.txt',
                            transform_param=dict(scale=1./256.),
                            shuffle=False,
                            ntop=2,
                            new_height=196,
                            new_width=196,
                            is_color=False)
    n.shift = input_shifting_5x5(n.input, batch_size)

    # Network
    n.dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    n.input_crop = L.Crop(n.input, n.dum_input, crop_param=dict(axis=2, offset=2))   
    n.conv1, n.poo1 = conv_conv_downsample_layer(n.input_crop, 3, 16, 2, 1)
    n.conv2, n.poo2 = conv_conv_downsample_layer(n.poo1, 3, 32, 2, 1)
    n.conv3, n.poo3 = conv_conv_downsample_layer(n.poo2, 3, 64, 2, 1)
    n.conv4, n.poo4 = conv_conv_downsample_layer(n.poo3, 3, 128, 2, 1)
    n.conv5, n.poo5 = conv_conv_downsample_layer(n.poo4, 3, 256, 2, 1)

    n.feature = conv_conv_layer(n.poo5, 3, 512, 1, 1, 0.5)

    n.deconv5 = upsample_concat_layer(n.feature, n.conv5, 3, 256, 2, 0, batch_size, 12)
    n.conv6 = conv_conv_layer(n.deconv5, 3, 256, 1, 1, 0.5)
    n.deconv6 = upsample_concat_layer(n.conv6, n.conv4, 3, 128, 2, 0, batch_size, 24)
    n.conv7 = conv_conv_layer(n.deconv6, 3, 128, 1, 1, 0.5)
    n.deconv7 = upsample_concat_layer(n.conv7, n.conv3, 3, 64, 2, 0, batch_size, 48)
    n.conv8 = conv_conv_layer(n.deconv7, 3, 64, 1, 1, 0.5)
    n.deconv8 = upsample_concat_layer(n.conv8, n.conv2, 3, 64, 2, 0, batch_size, 96)
    n.conv9 = conv_conv_layer(n.deconv8, 3, 64, 1, 1, 0.5)
    n.deconv9 = upsample_concat_layer(n.conv9, n.conv1, 3, 64, 2, 0, batch_size, 192)
    n.conv10 = conv_conv_layer(n.deconv9, 3, 64, 1, 1, 0.5)

    n.flow = flow_layer(n.conv10, 25*2)
    
    # Translation
    n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.predict = slice_warp(n.shift, n.flow_h, n.flow_v)

    n.edge = L.Python(n.predict, module='edge_layer', layer='EdgeLayer', ntop=1)

    # Loss
    n.label, n.trash = image_data_5x5(batch_size, 'train_source')

    n.dum_predict = L.DummyData(shape=dict(dim=[batch_size, 1, 180, 180]))
    n.predict_crop = L.Crop(n.predict, n.dum_predict, crop_param=dict(axis=2, offset=6))
    n.dum_label = L.DummyData(shape=dict(dim=[batch_size, 1, 180, 180]))
    n.label_crop = L.Crop(n.label, n.dum_label, crop_param=dict(axis=2, offset=6))     
    n.loss1 = L.AbsLoss(n.predict_crop, n.label_crop, loss_weight=1)

    n.dum_predict2 = L.DummyData(shape=dict(dim=[batch_size, 1, 180, 180]))
    n.edge_crop = L.Crop(n.edge, n.dum_predict2, crop_param=dict(axis=2, offset=6))
    n.edge_abs = L.AbsLoss(n.edge_crop, n.label_crop)
    n.loss2 = L.Power(n.edge_abs, power=1.0, scale=0.01, shift=0.0, in_place=False, loss_weight=1)

    return n.to_proto()

def denseUNet_test(batch_size=1):
    n = caffe.NetSpec()

    # Data loading
    n.input, n.trash = L.ImageData(batch_size=batch_size,
                            source='./datas/flower_dataset/train_source'+str(CENTER_SAI)+'.txt',
                            transform_param=dict(scale=1./256.),
                            shuffle=False,
                            ntop=2,
                            new_height=196,
                            new_width=196,
                            is_color=False)
    n.shift = input_shifting_5x5(n.input, batch_size)

    n.dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    n.input_crop = L.Crop(n.input, n.dum_input, crop_param=dict(axis=2, offset=2))    
    # Network
    n.conv1, n.poo1 = conv_conv_downsample_layer(n.input_crop, 3, 16, 2, 1)
    n.conv2, n.poo2 = conv_conv_downsample_layer(n.poo1, 3, 32, 2, 1)
    n.conv3, n.poo3 = conv_conv_downsample_layer(n.poo2, 3, 64, 2, 1)
    n.conv4, n.poo4 = conv_conv_downsample_layer(n.poo3, 3, 128, 2, 1)
    n.conv5, n.poo5 = conv_conv_downsample_layer(n.poo4, 3, 256, 2, 1)

    n.feature = conv_conv_layer(n.poo5, 3, 512, 1, 1, 0.0)
    
    n.deconv5 = upsample_concat_layer(n.feature, n.conv5, 3, 256, 2, 0, batch_size, 12)
    n.conv6 = conv_conv_layer(n.deconv5, 3, 256, 1, 1, 0.0)
    n.deconv6 = upsample_concat_layer(n.conv6, n.conv4, 3, 128, 2, 0, batch_size, 24)
    n.conv7 = conv_conv_layer(n.deconv6, 3, 128, 1, 1, 0.0)
    n.deconv7 = upsample_concat_layer(n.conv7, n.conv3, 3, 64, 2, 0, batch_size, 48)
    n.conv8 = conv_conv_layer(n.deconv7, 3, 64, 1, 1, 0.0)
    n.deconv8 = upsample_concat_layer(n.conv8, n.conv2, 3, 64, 2, 0, batch_size, 96)
    n.conv9 = conv_conv_layer(n.deconv8, 3, 64, 1, 1, 0.0)
    n.deconv9 = upsample_concat_layer(n.conv9, n.conv1, 3, 64, 2, 0, batch_size, 192)
    n.conv10 = conv_conv_layer(n.deconv9, 3, 64, 1, 1, 0.0)

    n.flow = flow_layer(n.conv10, 25*2)

    # Translation
    # n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    # n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    # n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer', layer = 'BilinearSamplerLayer', ntop = 1)

    n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.predict = slice_warp(n.shift, n.flow_h, n.flow_v)

    # Visualization
    n.label, n.trash = image_data_5x5(batch_size, 'train_source')
    n.trash1 = L.Python(n.flow_h, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/flower_dataset', name='flow_h', mult=30)))
    n.trash2 = L.Python(n.flow_v, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/flower_dataset', name='flow_v', mult=30)))
    n.trash3 = L.Python(n.shift, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/flower_dataset', name='input', mult=1*256)))
    n.trash4 = L.Python(n.label, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/flower_dataset', name='label', mult=1*256)))
    n.trash5 = L.Python(n.predict, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/flower_dataset', name='predict', mult=1*256)))

    return n.to_proto()
1
def denseUNet_solver(train_net_path, test_net_path=None, snapshot_path=None):
    s = caffe_pb2.SolverParameter()

    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 18
        s.test_iter.append(1)
    else:
        s.test_initialization = False
    
    s.iter_size = 1
    s.max_iter = 500000
    s.type = 'SGD'
    s.base_lr = 0.00005 # 0.0005,  0.0000001 # 0.000005(basic), 0.0000001
    s.lr_policy = 'fixed'
    s.gamma = 0.75
    s.power = 0.75
    s.stepsize = 1000
    s.momentum = 0.9
    s.momentum2 = 0.999
    # s.weight_decay = 0.0000005
    s.clip_gradients = 1
    s.display = 1

    s.snapshot = 500
    if snapshot_path is not None:
        s.snapshot_prefix = snapshot_path

    s.solver_mode = caffe_pb2.SolverParameter.GPU

    return s

if __name__ == "__main__":
    MODEL_PATH = './models'
    TRAIN_PATH = './scripts/denseUNet_train.prototxt'
    TEST_PATH = './scripts/denseUNet_test.prototxt'
    SOLVER_PATH = './scripts/denseUNet_solver.prototxt'

    def generate_net():
        with open(TRAIN_PATH, 'w') as f:
            f.write(str(denseUNet_train(6)))    
        with open(TEST_PATH, 'w') as f:
            f.write(str(denseUNet_test(1)))
    
    def generate_solver():
        with open(SOLVER_PATH, 'w') as f:
            f.write(str(denseUNet_solver(TRAIN_PATH, TEST_PATH, MODEL_PATH)))

    generate_net()
    generate_solver()
    solver = caffe.get_solver(SOLVER_PATH)
    #solver.net.copy_from('./models/denseUNet_solver_iter_20000.caffemodel')
    solver.solve()