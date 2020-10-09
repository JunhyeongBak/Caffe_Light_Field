# This program generates denseUNet's deploy prototxt and run it

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
import cv2
import os.path
import time
import numpy as np
import argparse
import caffe
import skimage
import json
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import math
#from skimage.measure import compare_ssim as ssim

SHIFT_VAL = 0.8

AF_INPLACE = True
BIAS_TERM = True

caffe.set_mode_cpu()
caffe.set_device(0)

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

def trans_SAIs_to_LF(imgSAIs):
    imgLF = np.zeros((192*1*5, 192*1*5, 3))
    full_LF_crop = np.zeros((192, 192, 3, 5, 5))
    for ax in range(5):
        for ay in range(5):
            full_LF_crop[:, :, :, ax, ay] = imgSAIs[:, :, :, 5*ax+ay]
    for ax in range(5):
        for ay in range(5):
            resized2 = full_LF_crop[:, :, :, ay, ax]
            resized2 = cv2.resize(resized2, dsize=(192*1, 192*1), interpolation=cv2.INTER_CUBIC) 
            imgLF[ay::5, ax::5, :] = resized2
    return imgLF

def trans_order(imgSAIs):
    imgSAIs2 = np.zeros((imgSAIs.shape))

    for i in range(25):
        if i == 0 or (i)%5==0:
            imgSAIs2[:, :, :, i//5] = imgSAIs[:, :, :, i]
        elif i == 1 or (i-1)%5==0:
            imgSAIs2[:, :, :, (i-1)//5+5] = imgSAIs[:, :, :, i]
        elif i == 2 or (i-2)%5==0:
            imgSAIs2[:, :, :, (i-2)//5+5*2] = imgSAIs[:, :, :, i]
        elif i == 3 or (i-3)%5==0:
            imgSAIs2[:, :, :, (i-3)//5+5*3] = imgSAIs[:, :, :, i]
        elif i == 4 or (i-4)%5==0:
            imgSAIs2[:, :, :, (i-4)//5+5*4] = imgSAIs[:, :, :, i]
    return imgSAIs2

def index_picker_8x8(i):
    i_list_5x5 = [9, 10, 11, 12, 13,
                    17, 18, 19, 20, 21,
                    25, 26, 27, 28, 29,
                    33, 34, 35, 36, 37,
                    41, 42, 43, 44, 45]
    i_pick = i_list_5x5[i]
    return i_pick

def flow_layer(bottom=None, nout=0):
    conv = L.Convolution(bottom, num_output=nout, kernel_size=3, stride=1, pad=1, dilation=1,
                            bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'), name='flow')
    return conv

def conv_conv_layer(bottom=None, nout=1, ks=3, strd=1, pad=1, dil=1):
    conv = L.Convolution(bottom, num_output=nout, kernel_size=ks, stride=strd, pad=pad, dilation=dil,
                            bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    conv = L.Convolution(conv, num_output=nout, kernel_size=ks, stride=strd, pad=pad, dilation=dil,
                            bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    return conv

def conv_conv_group_layer(bottom=None, nout=1, ks=3, strd=1, pad=1, dil=1):
    conv = L.Convolution(bottom, num_output=nout, kernel_size=ks, stride=strd, pad=pad,
                            bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    conv = L.Convolution(conv, num_output=nout, kernel_size=ks, stride=strd, pad=pad,
                            group=nout, bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.Convolution(conv, num_output=nout, kernel_size=1, stride=strd, pad=0,
                            bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    return conv

def conv_conv_res_dense_layer(bottom=None, nout=1, ks=3, strd=1, pad=1, dil=1):
    bottom = L.Convolution(bottom, num_output=nout, kernel_size=ks, stride=strd, pad=pad, dilation=dil,
                            bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    bottom = L.ReLU(bottom, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    conv = L.Convolution(bottom, num_output=nout, kernel_size=ks, stride=strd, pad=pad+(dil*3-1), dilation=dil*3,
                            bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    conv = L.Convolution(conv, num_output=nout, kernel_size=ks, stride=strd, pad=pad+(dil*6-1), dilation=dil*6,
                            bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    elth = L.Eltwise(conv, bottom, operation=P.Eltwise.SUM)
    elth = L.ReLU(elth, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    return elth   

def conv_conv_downsample_layer(bottom=None, nout=1, ks=3, strd=2, pad=0):
    conv = L.Convolution(bottom, num_output=nout, kernel_size=ks, stride=1, pad=pad, dilation=1,
                            bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    conv = L.Convolution(conv, num_output=nout, kernel_size=ks, stride=1, pad=pad,
                            bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    pool = L.Pooling(conv, kernel_size=ks, stride=strd, pool=P.Pooling.MAX) ##
    pool = L.ReLU(pool, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    return conv, pool

def conv_conv_downsample_group_layer(bottom=None, nout=1, ks=3, strd=2, pad=0):
    conv = L.Convolution(bottom, num_output=nout, kernel_size=ks, stride=1, pad=pad,
                            bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    conv = L.Convolution(conv, num_output=nout, kernel_size=ks, stride=1, pad=pad,
                            group=nout, bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.Convolution(conv, num_output=nout, kernel_size=1, stride=1, pad=0,
                            bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    pool = L.Pooling(conv, kernel_size=ks, stride=strd, pool=P.Pooling.MAX) ##
    pool = L.ReLU(pool, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    return conv, pool

def upsample_concat_layer(bottom1=None, bottom2=None, nout=1, ks=3, strd=2, pad=0, batch_size=1, crop_size=0):
    deconv = L.Deconvolution(bottom1, convolution_param=dict(num_output=nout, kernel_size=ks, stride=strd, pad=pad))
    deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    dum = L.DummyData(shape=dict(dim=[batch_size, nout, crop_size, crop_size]))
    deconv_crop = L.Crop(deconv, dum, crop_param=dict(axis=2, offset=0))
    conc = L.Concat(*[deconv_crop, bottom2], concat_param={'axis': 1})
    return conc

def input_shifting(center, batch_size, grid_size):
    con = None
    for i in range(grid_size):
        if grid_size == 64:
            tx, ty = shift_value_8x8(i, SHIFT_VAL)
        elif grid_size == 25:
            i_pick = index_picker_8x8(i)
            tx, ty = shift_value_8x8(i_pick, SHIFT_VAL)
        else:
            print('This size is not supported')
        param_str = json.dumps({'tx': tx, 'ty': ty})
        shift = L.Python(center, module = 'input_shifting_layer', layer = 'InputShiftingLayer', ntop = 1, param_str = param_str)
        if i == 0:
            con = shift
        else:   
            con = L.Concat(*[con, shift], concat_param={'axis': 1})
    #dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    #con_crop = L.Crop(con, dum_input, crop_param=dict(axis=2, offset=2))
    return con

def denseUNet_deploy_model0():
    batch_size = 1
    n = caffe.NetSpec()

    # Data loading
    n.input = L.Input(input_param=dict(shape=dict(dim=[1, 3, 192, 192])))
    n.input_luma = L.Input(input_param=dict(shape=dict(dim=[1, 1, 192, 192])))
    n.input_luma_down = L.Power(n.input_luma, power=1.0, scale=1./256., shift=0, in_place=False)
    n.shift = input_shifting(n.input, batch_size, 25)

    # Network
    #n.dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    #n.input_crop = L.Crop(n.input, n.dum_input, crop_param=dict(axis=2, offset=2))   
    n.conv1_1, n.poo1 = conv_conv_downsample_layer(n.input_luma_down, 16, 3, 2, 1)
    n.conv2_1, n.poo2 = conv_conv_downsample_layer(n.poo1, 32, 3, 2, 1)
    n.conv3_1, n.poo3 = conv_conv_downsample_layer(n.poo2, 64, 3, 2, 1)
    n.conv4_1, n.poo4 = conv_conv_downsample_layer(n.poo3, 128, 3, 2, 1)
    n.conv5_1, n.poo5 = conv_conv_downsample_layer(n.poo4, 256, 3, 2, 1)
    n.feature = conv_conv_res_dense_layer(n.poo5, 512, 3, 1, 1, 1)

    n.deconv1 = upsample_concat_layer(n.feature, n.conv5_1, 256, 3, 2, 0, batch_size, 12)
    n.deconv1 = conv_conv_layer(n.deconv1, 256, 3, 1, 1, 1)

    n.deconv2 = upsample_concat_layer(n.deconv1, n.conv4_1, 128, 3, 2, 0, batch_size, 24)
    n.deconv2 = conv_conv_layer(n.deconv2, 128, 3, 1, 1, 1)

    n.deconv3 = upsample_concat_layer(n.deconv2, n.conv3_1, 64, 3, 2, 0, batch_size, 48)
    n.deconv3 = conv_conv_layer(n.deconv3, 64, 3, 1, 1, 1)

    n.deconv4 = upsample_concat_layer(n.deconv3, n.conv2_1, 64, 3, 2, 0, batch_size, 96)
    n.deconv4 = conv_conv_layer(n.deconv4, 64, 3, 1, 1, 1)

    n.deconv5 = upsample_concat_layer(n.deconv4, n.conv1_1, 64, 3, 2, 0, batch_size, 192)
    n.deconv5 = conv_conv_layer(n.deconv5, 64, 3, 1, 1, 1)

    n.flow25 = flow_layer(n.deconv5, 25*2)


    # Translation
    n.flow_h, n.flow_v = L.Slice(n.flow25, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    #n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1)
    param_str = json.dumps({'flow_size': 25, 'color_size': 3})
    n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1, param_str = param_str)

    '''
    n.trash1 = L.Python(n.flow_h, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./deploy', name='flow_h', mult=110)))
    n.trash2 = L.Python(n.flow_v, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./deploy', name='flow_v', mult=110)))
    '''

    return n.to_proto()

def denseUNet_deploy_model1():
    batch_size = 1
    n = caffe.NetSpec()

    # Data loading
    n.input = L.Input(input_param=dict(shape=dict(dim=[1, 3, 192, 192])))
    n.input_luma = L.Input(input_param=dict(shape=dict(dim=[1, 1, 192, 192])))
    n.input_luma_down = L.Power(n.input_luma, power=1.0, scale=1./256., shift=0, in_place=False)
    n.shift = input_shifting(n.input, batch_size, 25)

    # Network
    #n.dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    #n.input_crop = L.Crop(n.input, n.dum_input, crop_param=dict(axis=2, offset=2))   
    n.conv1_1, n.poo1 = conv_conv_downsample_layer(n.input_luma_down, 16, 3, 2, 1)
    n.conv2_1, n.poo2 = conv_conv_downsample_layer(n.poo1, 32, 3, 2, 1)
    n.conv3_1, n.poo3 = conv_conv_downsample_layer(n.poo2, 64, 3, 2, 1)
    n.conv4_1, n.poo4 = conv_conv_downsample_layer(n.poo3, 128, 3, 2, 1)
    n.feature = conv_conv_res_dense_layer(n.poo4, 256, 3, 1, 1, 1)

    n.deconv1 = upsample_concat_layer(n.feature, n.conv4_1, 128, 3, 2, 0, batch_size, 24)
    n.deconv1 = conv_conv_layer(n.deconv1, 128, 3, 1, 1, 1)

    n.deconv2 = upsample_concat_layer(n.deconv1, n.conv3_1, 64, 3, 2, 0, batch_size, 48)
    n.deconv2 = conv_conv_layer(n.deconv2, 64, 3, 1, 1, 1)

    n.deconv3 = upsample_concat_layer(n.deconv2, n.conv2_1, 64, 3, 2, 0, batch_size, 96)
    n.deconv3 = conv_conv_layer(n.deconv3, 64, 3, 1, 1, 1)

    n.deconv4 = upsample_concat_layer(n.deconv3, n.conv1_1, 64, 3, 2, 0, batch_size, 192)
    n.deconv4 = conv_conv_layer(n.deconv4, 64, 3, 1, 1, 1)

    n.flow25 = flow_layer(n.deconv4, 25*2)


    # Translation
    n.flow_h, n.flow_v = L.Slice(n.flow25, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    #n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1)
    param_str = json.dumps({'flow_size': 25, 'color_size': 3})
    n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1, param_str = param_str)

    '''
    n.trash1 = L.Python(n.flow_h, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./deploy', name='flow_h', mult=110)))
    n.trash2 = L.Python(n.flow_v, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./deploy', name='flow_v', mult=110)))
    '''

    return n.to_proto()

def denseUNet_deploy_model2():
    batch_size = 1
    n = caffe.NetSpec()

    # Data loading
    n.input = L.Input(input_param=dict(shape=dict(dim=[1, 3, 192, 192])))
    n.input_luma = L.Input(input_param=dict(shape=dict(dim=[1, 1, 192, 192])))
    n.input_luma_down = L.Power(n.input_luma, power=1.0, scale=1./256., shift=0, in_place=False)
    n.shift = input_shifting(n.input, batch_size, 25)

    # Network
    #n.dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    #n.input_crop = L.Crop(n.input, n.dum_input, crop_param=dict(axis=2, offset=2))   
    n.conv1_1, n.poo1 = conv_conv_downsample_layer(n.input_luma_down, 16, 3, 2, 1)
    n.conv2_1, n.poo2 = conv_conv_downsample_layer(n.poo1, 32, 3, 2, 1)
    n.conv3_1, n.poo3 = conv_conv_downsample_layer(n.poo2, 64, 3, 2, 1)
    n.conv4_1, n.poo4 = conv_conv_downsample_layer(n.poo3, 128, 3, 2, 1)
    n.feature = conv_conv_res_dense_layer(n.poo4, 256, 3, 1, 1, 1)

    n.deconv1 = upsample_concat_layer(n.feature, n.conv4_1, 128, 3, 2, 0, batch_size, 24)
    n.deconv1 = conv_conv_layer(n.deconv1, 128, 3, 1, 1, 1)

    n.conv3_1 = upsample_concat_layer(n.conv4_1, n.conv3_1, 64, 3, 2, 0, batch_size, 48)
    n.conv3_2 = conv_conv_layer(n.conv3_1, 64, 3, 1, 1, 1)
    n.conv3_2 = L.Concat(*[n.conv3_2, n.conv3_1], concat_param={'axis': 1})
    n.deconv2 = upsample_concat_layer(n.deconv1, n.conv3_2, 64, 3, 2, 0, batch_size, 48)
    n.deconv2 = conv_conv_layer(n.deconv2, 64, 3, 1, 1, 1)

    n.conv2_1 = upsample_concat_layer(n.conv3_1, n.conv2_1, 64, 3, 2, 0, batch_size, 96)
    n.conv2_2 = conv_conv_layer(n.conv2_1, 64, 3, 1, 1, 1)
    n.conv2_2 = L.Concat(*[n.conv2_2, n.conv2_1], concat_param={'axis': 1})
    n.conv2_2 = upsample_concat_layer(n.conv3_2, n.conv2_2, 64, 3, 2, 0, batch_size, 96)
    n.conv2_3 = conv_conv_layer(n.conv2_2, 64, 3, 1, 1, 1)
    n.conv2_3 = L.Concat(*[n.conv2_3, n.conv2_2, n.conv2_1], concat_param={'axis': 1})
    n.deconv3 = upsample_concat_layer(n.deconv2, n.conv2_3, 64, 3, 2, 0, batch_size, 96)
    n.deconv3 = conv_conv_layer(n.deconv3, 64, 3, 1, 1, 1)

    n.conv1_1 = upsample_concat_layer(n.conv2_1, n.conv1_1, 64, 3, 2, 0, batch_size, 192)
    n.conv1_2 = conv_conv_layer(n.conv1_1, 64, 3, 1, 1, 1)
    n.conv1_2 = L.Concat(*[n.conv1_2, n.conv1_1], concat_param={'axis': 1})
    n.conv1_2 = upsample_concat_layer(n.conv2_2, n.conv1_2, 64, 3, 2, 0, batch_size, 192)
    n.conv1_3 = conv_conv_layer(n.conv1_2, 64, 3, 1, 1, 1)
    n.conv1_3 = L.Concat(*[n.conv1_3, n.conv1_2, n.conv1_1], concat_param={'axis': 1})
    n.conv1_3 = upsample_concat_layer(n.conv2_3, n.conv1_3, 64, 3, 2, 0, batch_size, 192)
    n.conv1_4 = conv_conv_layer(n.conv1_3, 64, 3, 1, 1, 1)
    n.conv1_4 = L.Concat(*[n.conv1_4, n.conv1_3, n.conv1_2, n.conv1_1], concat_param={'axis': 1})
    n.deconv4 = upsample_concat_layer(n.deconv3, n.conv1_4, 64, 3, 2, 0, batch_size, 192)
    n.deconv4 = conv_conv_layer(n.deconv4, 64, 3, 1, 1, 1)

    n.flow25 = flow_layer(n.deconv4, 25*2)


    # Translation
    n.flow_h, n.flow_v = L.Slice(n.flow25, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    #n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1)
    param_str = json.dumps({'flow_size': 25, 'color_size': 3})
    n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1, param_str = param_str)

    '''
    n.trash1 = L.Python(n.flow_h, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./deploy', name='flow_h', mult=110)))
    n.trash2 = L.Python(n.flow_v, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./deploy', name='flow_v', mult=110)))
    '''

    return n.to_proto()

def denseUNet_deploy_model3():
    batch_size = 1
    n = caffe.NetSpec()

    # Data loading
    n.input = L.Input(input_param=dict(shape=dict(dim=[1, 3, 192, 192])))
    n.input_luma = L.Input(input_param=dict(shape=dict(dim=[1, 1, 192, 192])))
    n.input_luma_down = L.Power(n.input_luma, power=1.0, scale=1./256., shift=0, in_place=False)
    n.shift = input_shifting(n.input, batch_size, 25)

    # Network
    #n.dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    #n.input_crop = L.Crop(n.input, n.dum_input, crop_param=dict(axis=2, offset=2))   
    n.conv1_1, n.poo1 = conv_conv_downsample_layer(n.input_luma_down, 16, 3, 2, 1)
    n.conv2_1, n.poo2 = conv_conv_downsample_layer(n.poo1, 32, 3, 2, 1)
    n.conv3_1, n.poo3 = conv_conv_downsample_layer(n.poo2, 64, 3, 2, 1)
    n.conv4_1, n.poo4 = conv_conv_downsample_layer(n.poo3, 128, 3, 2, 1)
    n.feature = conv_conv_res_dense_layer(n.poo4, 256, 3, 1, 1, 1)

    n.deconv1 = upsample_concat_layer(n.feature, n.conv4_1, 128, 3, 2, 0, batch_size, 24)
    n.deconv1 = conv_conv_layer(n.deconv1, 128, 3, 1, 1, 1)

    n.conv3_2 = conv_conv_layer(n.conv3_1, 64, 3, 1, 1, 1)
    n.conv3_2 = L.Concat(*[n.conv3_2, n.conv3_1], concat_param={'axis': 1})
    n.deconv2 = upsample_concat_layer(n.deconv1, n.conv3_2, 64, 3, 2, 0, batch_size, 48)
    n.deconv2 = conv_conv_layer(n.deconv2, 64, 3, 1, 1, 1)

    n.conv2_2 = conv_conv_layer(n.conv2_1, 64, 3, 1, 1, 1)
    n.conv2_3 = conv_conv_layer(n.conv2_2, 64, 3, 1, 1, 1)
    n.conv2_3 = L.Concat(*[n.conv2_3, n.conv2_2, n.conv2_1], concat_param={'axis': 1})
    n.deconv3 = upsample_concat_layer(n.deconv2, n.conv2_3, 64, 3, 2, 0, batch_size, 96)
    n.deconv3 = conv_conv_layer(n.deconv3, 64, 3, 1, 1, 1)

    n.conv1_2 = conv_conv_layer(n.conv1_1, 64, 3, 1, 1, 1)
    n.conv1_3 = conv_conv_layer(n.conv1_2, 64, 3, 1, 1, 1)
    n.conv1_4 = conv_conv_layer(n.conv1_3, 64, 3, 1, 1, 1)
    n.conv1_4 = L.Concat(*[n.conv1_4, n.conv1_3, n.conv1_2, n.conv1_1], concat_param={'axis': 1})
    n.deconv4 = upsample_concat_layer(n.deconv3, n.conv1_4, 64, 3, 2, 0, batch_size, 192)
    n.deconv4 = conv_conv_layer(n.deconv4, 64, 3, 1, 1, 1)

    n.flow25 = flow_layer(n.deconv4, 25*2)


    # Translation
    n.flow_h, n.flow_v = L.Slice(n.flow25, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    #n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1)
    param_str = json.dumps({'flow_size': 25, 'color_size': 3})
    n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1, param_str = param_str)

    '''
    n.trash1 = L.Python(n.flow_h, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./deploy', name='flow_h', mult=110)))
    n.trash2 = L.Python(n.flow_v, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./deploy', name='flow_v', mult=110)))
    '''

    return n.to_proto()

def denseUNet_deploy_model4():
    batch_size = 1
    n = caffe.NetSpec()

    # Data loading
    n.input = L.Input(input_param=dict(shape=dict(dim=[1, 3, 192, 192])))
    n.input_luma = L.Input(input_param=dict(shape=dict(dim=[1, 1, 192, 192])))
    n.input_luma_down = L.Power(n.input_luma, power=1.0, scale=1./256., shift=0, in_place=False)
    n.shift = input_shifting(n.input, batch_size, 25)

    # Network
    #n.dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    #n.input_crop = L.Crop(n.input, n.dum_input, crop_param=dict(axis=2, offset=2))   
    n.conv1_1, n.poo1 = conv_conv_downsample_layer(n.input_luma_down, 16, 3, 2, 1)
    n.conv2_1, n.poo2 = conv_conv_downsample_layer(n.poo1, 32, 3, 2, 1)
    n.conv3_1, n.poo3 = conv_conv_downsample_layer(n.poo2, 64, 3, 2, 1)
    n.conv4_1, n.poo4 = conv_conv_downsample_layer(n.poo3, 128, 3, 2, 1)
    n.feature = conv_conv_res_dense_layer(n.poo4, 256, 3, 1, 1, 1)

    n.deconv1 = upsample_concat_layer(n.feature, n.conv4_1, 128, 3, 2, 0, batch_size, 24)
    n.deconv1 = conv_conv_layer(n.deconv1, 128, 3, 1, 1, 1)

    n.conv3_2 = conv_conv_layer(n.conv3_1, 64, 3, 1, 1, 1)
    n.conv3_2 = L.Concat(*[n.conv3_2, n.conv3_1], concat_param={'axis': 1})
    n.deconv2 = upsample_concat_layer(n.deconv1, n.conv3_2, 64, 3, 2, 0, batch_size, 48)
    n.deconv2 = conv_conv_layer(n.deconv2, 64, 3, 1, 1, 1)

    n.conv2_2 = conv_conv_layer(n.conv2_1, 64, 3, 1, 1, 1)
    n.conv2_2 = L.Concat(*[n.conv2_2, n.conv2_1], concat_param={'axis': 1})
    n.conv2_3 = conv_conv_layer(n.conv2_2, 64, 3, 1, 1, 1)
    n.conv2_3 = L.Concat(*[n.conv2_3, n.conv2_2, n.conv2_1], concat_param={'axis': 1})
    n.deconv3 = upsample_concat_layer(n.deconv2, n.conv2_3, 64, 3, 2, 0, batch_size, 96)
    n.deconv3 = conv_conv_layer(n.deconv3, 64, 3, 1, 1, 1)

    n.conv1_2 = conv_conv_layer(n.conv1_1, 64, 3, 1, 1, 1)
    n.conv1_2 = L.Concat(*[n.conv1_2, n.conv1_1], concat_param={'axis': 1})
    n.conv1_3 = conv_conv_layer(n.conv1_2, 64, 3, 1, 1, 1)
    n.conv1_3 = L.Concat(*[n.conv1_3, n.conv1_2, n.conv1_1], concat_param={'axis': 1})
    n.conv1_4 = conv_conv_layer(n.conv1_3, 64, 3, 1, 1, 1)
    n.conv1_4 = L.Concat(*[n.conv1_4, n.conv1_3, n.conv1_2, n.conv1_1], concat_param={'axis': 1})
    n.deconv4 = upsample_concat_layer(n.deconv3, n.conv1_4, 64, 3, 2, 0, batch_size, 192)
    n.deconv4 = conv_conv_layer(n.deconv4, 64, 3, 1, 1, 1)

    n.flow25 = flow_layer(n.deconv4, 25*2)


    # Translation
    n.flow_h, n.flow_v = L.Slice(n.flow25, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    #n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1)
    param_str = json.dumps({'flow_size': 25, 'color_size': 3})
    n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1, param_str = param_str)

    '''
    n.trash1 = L.Python(n.flow_h, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./deploy', name='flow_h', mult=110)))
    n.trash2 = L.Python(n.flow_v, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./deploy', name='flow_v', mult=110)))
    '''

    return n.to_proto()

def denseUNet_deploy_model5():
    batch_size = 1
    n = caffe.NetSpec()

    # Data loading
    n.input = L.Input(input_param=dict(shape=dict(dim=[1, 3, 192, 192])))
    n.input_luma = L.Input(input_param=dict(shape=dict(dim=[1, 1, 192, 192])))
    n.input_luma_down = L.Power(n.input_luma, power=1.0, scale=1./256., shift=0, in_place=False)
    n.shift = input_shifting(n.input, batch_size, 25)

    # Network
    #n.dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    #n.input_crop = L.Crop(n.input, n.dum_input, crop_param=dict(axis=2, offset=2))   
    n.conv1_1, n.poo1 = conv_conv_downsample_layer(n.input_luma_down, 16, 3, 2, 1)
    n.conv2_1, n.poo2 = conv_conv_downsample_layer(n.poo1, 32, 3, 2, 1)
    n.conv3_1, n.poo3 = conv_conv_downsample_layer(n.poo2, 64, 3, 2, 1)
    n.conv4_1, n.poo4 = conv_conv_downsample_layer(n.poo3, 128, 3, 2, 1)
    n.feature = conv_conv_res_dense_layer(n.poo4, 256, 3, 1, 1, 1)

    n.deconv1 = upsample_concat_layer(n.feature, n.conv4_1, 128, 3, 2, 0, batch_size, 24)
    n.deconv1 = conv_conv_layer(n.deconv1, 128, 3, 1, 1, 1)

    n.conv3_1 = upsample_concat_layer(n.conv4_1, n.conv3_1, 64, 3, 2, 0, batch_size, 48)
    n.conv3_2 = conv_conv_layer(n.conv3_1, 64, 3, 1, 1, 1)
    n.conv3_2 = L.Concat(*[n.conv3_2, n.conv3_1], concat_param={'axis': 1})
    n.deconv2 = upsample_concat_layer(n.deconv1, n.conv3_2, 64, 3, 2, 0, batch_size, 48)
    n.deconv2 = conv_conv_layer(n.deconv2, 64, 3, 1, 1, 1)

    n.conv2_1 = upsample_concat_layer(n.conv3_1, n.conv2_1, 64, 3, 2, 0, batch_size, 96)
    n.conv2_2 = conv_conv_layer(n.conv2_1, 64, 3, 1, 1, 1)
    n.conv2_2 = L.Concat(*[n.conv2_2, n.conv2_1], concat_param={'axis': 1})
    n.conv2_2 = upsample_concat_layer(n.conv3_2, n.conv2_2, 64, 3, 2, 0, batch_size, 96)
    n.conv2_3 = conv_conv_layer(n.conv2_2, 64, 3, 1, 1, 1)
    n.conv2_3 = L.Concat(*[n.conv2_3, n.conv2_2, n.conv2_1], concat_param={'axis': 1})
    n.deconv3 = upsample_concat_layer(n.deconv2, n.conv2_3, 64, 3, 2, 0, batch_size, 96)
    n.deconv3 = conv_conv_layer(n.deconv3, 64, 3, 1, 1, 1)

    n.conv1_1 = upsample_concat_layer(n.conv2_1, n.conv1_1, 64, 3, 2, 0, batch_size, 192)
    n.conv1_2 = conv_conv_layer(n.conv1_1, 64, 3, 1, 1, 1)
    n.conv1_2 = L.Concat(*[n.conv1_2, n.conv1_1], concat_param={'axis': 1})
    n.conv1_2 = upsample_concat_layer(n.conv2_2, n.conv1_2, 64, 3, 2, 0, batch_size, 192)
    n.conv1_3 = conv_conv_layer(n.conv1_2, 64, 3, 1, 1, 1)
    n.conv1_3 = L.Concat(*[n.conv1_3, n.conv1_2, n.conv1_1], concat_param={'axis': 1})
    n.conv1_3 = upsample_concat_layer(n.conv2_3, n.conv1_3, 64, 3, 2, 0, batch_size, 192)
    n.conv1_4 = conv_conv_layer(n.conv1_3, 64, 3, 1, 1, 1)
    n.conv1_4 = L.Concat(*[n.conv1_4, n.conv1_3, n.conv1_2, n.conv1_1], concat_param={'axis': 1})
    n.deconv4 = upsample_concat_layer(n.deconv3, n.conv1_4, 64, 3, 2, 0, batch_size, 192)
    n.deconv4 = conv_conv_layer(n.deconv4, 64, 3, 1, 1, 1)

    n.final_con = L.Concat(*[n.deconv4, n.conv1_4, n.conv1_3, n.conv1_2], concat_param={'axis': 1})

    n.flow25 = flow_layer(n.final_con, 25*2)


    # Translation
    n.flow_h, n.flow_v = L.Slice(n.flow25, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    #n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1)
    param_str = json.dumps({'flow_size': 25, 'color_size': 3})
    n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1, param_str = param_str)

    '''
    n.trash1 = L.Python(n.flow_h, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./deploy', name='flow_h', mult=110)))
    n.trash2 = L.Python(n.flow_v, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./deploy', name='flow_v', mult=110)))
    '''

    return n.to_proto()

def denseUNet_deploy_mobilenet():
    batch_size = 1
    n = caffe.NetSpec()

    # Data loading
    n.input = L.Input(input_param=dict(shape=dict(dim=[1, 3, 192, 192])))
    n.input_luma = L.Input(input_param=dict(shape=dict(dim=[1, 1, 192, 192])))
    n.input_luma_down = L.Power(n.input_luma, power=1.0, scale=1./256., shift=0, in_place=False)
    n.shift = input_shifting(n.input, batch_size, 25)

    # Network
    #n.dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    #n.input_crop = L.Crop(n.input, n.dum_input, crop_param=dict(axis=2, offset=2))   
    n.conv1_1, n.poo1 = conv_conv_downsample_group_layer(n.input_luma_down, 16, 3, 2, 1)
    n.conv2_1, n.poo2 = conv_conv_downsample_group_layer(n.poo1, 32, 3, 2, 1)
    n.conv3_1, n.poo3 = conv_conv_downsample_group_layer(n.poo2, 64, 3, 2, 1)
    n.conv4_1, n.poo4 = conv_conv_downsample_group_layer(n.poo3, 128, 3, 2, 1)
    n.feature = conv_conv_res_dense_layer(n.poo4, 256, 3, 1, 1, 1)

    n.deconv1 = upsample_concat_layer(n.feature, n.conv4_1, 128, 3, 2, 0, batch_size, 24)
    n.deconv1 = conv_conv_group_layer(n.deconv1, 128, 3, 1, 1, 1)

    n.deconv2 = upsample_concat_layer(n.deconv1, n.conv3_1, 64, 3, 2, 0, batch_size, 48)
    n.deconv2 = conv_conv_group_layer(n.deconv2, 64, 3, 1, 1, 1)

    n.deconv3 = upsample_concat_layer(n.deconv2, n.conv2_1, 64, 3, 2, 0, batch_size, 96)
    n.deconv3 = conv_conv_group_layer(n.deconv3, 64, 3, 1, 1, 1)

    n.deconv4 = upsample_concat_layer(n.deconv3, n.conv1_1, 64, 3, 2, 0, batch_size, 192)
    n.deconv4 = conv_conv_group_layer(n.deconv4, 64, 3, 1, 1, 1)

    n.flow25 = flow_layer(n.deconv4, 25*2)


    # Translation
    n.flow_h, n.flow_v = L.Slice(n.flow25, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    #n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1)
    param_str = json.dumps({'flow_size': 25, 'color_size': 3})
    n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1, param_str = param_str)

    '''
    n.trash1 = L.Python(n.flow_h, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./deploy', name='flow_h', mult=110)))
    n.trash2 = L.Python(n.flow_v, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./deploy', name='flow_v', mult=110)))
    '''

    return n.to_proto()

if __name__ == "__main__":
    TOT = 77
    PATH = './deploy'
    NET_PATH = PATH + '/denseUNet_deploy.prototxt'
    WEIGHTS_PATH = PATH + '/mdoel_mobilenet' + '/denseUNet_solver_iter_16535.caffemodel' # 3307, 6614, 9921, 13228, 16535

    # Generate a network
    def generate_net():
        with open(NET_PATH, 'w') as f:
            f.write(str(denseUNet_deploy_mobilenet()))
    generate_net()

    # Set a model
    net = caffe.Net(NET_PATH, WEIGHTS_PATH, caffe.TEST)

    psnr_mean = 0
    ssim_mean = 0
    time_mean = 0
    for i_tot in range(0, 1):
        start = time.time()

        # Input images
        src_color = cv2.imread(PATH + '/input_flower_8x8/sai'+str(i_tot)+'_27.png', cv2.IMREAD_COLOR)
        if not os.path.isfile(PATH + '/input_flower_8x8/sai'+str(i_tot)+'_27.png'):
            src_color = cv2.imread(PATH + '/input_flower_8x8/sai'+str(i_tot)+'_27.jpg', cv2.IMREAD_COLOR)
            if not os.path.isfile(PATH + '/input_flower_8x8/sai'+str(i_tot)+'_27.jpg'):
                raise Exception('(ERR) The image file is not exist!')
        
        src_color = cv2.resize(src_color, dsize=(192, 192), interpolation=cv2.INTER_AREA) ###
        src_luma = cv2.cvtColor(src_color, cv2.COLOR_BGR2GRAY)
        src_blob_color = np.zeros((1, 3, 192, 192))
        src_blob_luma = np.zeros((1, 1, 192, 192))
        for i in range(3):
            src_blob_color[:, i, :, :] = src_color[:, :, i]
        src_blob_luma[:, 0, :, :] = src_luma[:, :]
        net.blobs['input'].data[...] = src_blob_color
        net.blobs['input_luma'].data[...] = src_blob_luma

        # Net forward        
        res = net.forward()

        ########## Predict ##########

        # Get and print sai
        sai = np.zeros((192, 192, 3))
        sai_list =  np.zeros((192, 192, 3, 5*5))
        dst_blob = net.blobs['predict'].data[...]
        for i in range(25):
            dst_blob_slice = dst_blob[:, 3*i:3*i+3, :, :]
            for c in range(3):
                sai[:, :, c] = cv2.resize(dst_blob_slice[0, c, :, :], dsize=(192, 192), interpolation=cv2.INTER_AREA)
            sai_list[:, :, :, i] = sai
            cv2.imwrite(PATH+'/output/sai'+str(i_tot)+'_'+str(i)+'.png', sai)
        
        # Print lf
        sai_list2 = trans_order(sai_list)
        lf = trans_SAIs_to_LF(sai_list2)
        cv2.imwrite(PATH+'/output/result_lf'+str(i_tot)+'.jpg', lf)
        
        # Print grid and epi
        sai_uv = np.zeros((192, 192, 3, 5, 5))
        sai_grid = np.zeros((192*5, 192*5, 3))
        sai_epi_ver = np.zeros((192, 5*5, 3))
        sai_epi_hor = np.zeros((5*5, 192, 3))
        for ax in range(5):
            for ay in range(5):
                sai = sai_list[:, :, :, 5*ax+ay]
                sai_uv[:, :, :, ay, ax] = sai
                sai_grid[192*ay:192*ay+192, 192*ax:192*ax+192, :] = sai
                sai_epi_ver[:, 5*ax+ay, :] = sai_uv[:, 192//2, :, ay, ax]
                sai_epi_hor[5*ax+ay, :, :] = sai_uv[192//2, :, :, ay, ax]
        cv2.imwrite(PATH+'/output/sai_grid'+str(i_tot)+'.png', sai_grid)
        cv2.imwrite(PATH+'/output/sai_epi_ver'+str(i_tot)+'.png', sai_epi_ver)
        cv2.imwrite(PATH+'/output/sai_epi_hor'+str(i_tot)+'.png', sai_epi_hor)

        end = time.time()
        time_mean = time_mean + end-start
        print('i : '+str(i_tot)+' | '+'time labs : '+str(end-start))

    time_mean = time_mean / TOT
    print('Total result | '+'time labs : '+str(time_mean))
    exit(0)