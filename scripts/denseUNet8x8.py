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
                                num_output=nout, pad=1, bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'), name='flow'+str(nout//2)+'0')
    return conv

def conv_layer(bottom=None, ks=3, nout=1, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    return conv

def conv_final_layer(bottom=None, ks=3, nout=1, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.TanH(conv, in_place=AF_INPLACE)
    return conv

def conv_conv_layer(bottom=None, ks=3, nout=1, stride=1, pad=1, drop=0.5):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, dilation=1, bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    if drop != 0: 
        conv = L.Dropout(conv, dropout_param=dict(dropout_ratio=drop), in_place=False)
    conv = L.Convolution(conv, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    return conv

def downsample_layer(bottom=None, ks=3, stride=2):
    pool = L.Pooling(bottom, kernel_size=ks, stride=stride, pool=P.Pooling.MAX)
    pool = L.ReLU(pool, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    return pool

def upsample_layer(bottom=None, ks=2, nout=1, stride=2):
    deconv = L.Deconvolution(bottom, convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride, pad=0))
    deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    return deconv

def upsample_concat_layer(bottom1=None, bottom2=None, ks=2, nout=1, stride=2, pad=0, batch_size=1, crop_size=0):
    deconv = L.Deconvolution(bottom1, convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride, pad=pad))
    deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    dum = L.DummyData(shape=dict(dim=[batch_size, nout, crop_size, crop_size]))
    deconv_crop = L.Crop(deconv, dum, crop_param=dict(axis=2, offset=0))
    conc = L.Concat(*[deconv_crop, bottom2], concat_param={'axis': 1})
    return conc

def upsample_concat_crop_layer(bottom1=None, bottom2=None, ks=2, nout=1, stride=2, pad=0, batch_size=1, crop_size=0):
    deconv = L.Deconvolution(bottom1, convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride, pad=pad))
    deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    dum = L.DummyData(shape=dict(dim=[batch_size, 64, 192, 192]))
    deconv_crop = L.Crop(deconv, dum, crop_param=dict(axis=2, offset=0))
    conc = L.Concat(*[deconv_crop, bottom2], concat_param={'axis': 1})
    return conc

def upsample_conv_conv_layer(bottom=None, ks=3, nout=1, stride=2, pad=1):
    deconv = L.Deconvolution(bottom, convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride))
    deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    conv = L.Convolution(deconv, kernel_size=ks, stride=1,
                                num_output=nout, pad=pad, bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    conv = L.Convolution(conv, kernel_size=ks, stride=1,
                                num_output=nout, pad=pad, bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.TanH(conv, in_place=AF_INPLACE)
    return conv

def conv_conv_downsample_layer(bottom=None, ks=3, nout=1, stride=2, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=1,
                                num_output=nout, pad=pad, dilation=1, bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    conv = L.Convolution(conv, kernel_size=ks, stride=1,
                                num_output=nout, pad=pad, bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    pool = L.Pooling(conv, kernel_size=ks, stride=stride, pool=P.Pooling.MAX) ##
    pool = L.ReLU(pool, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    return conv, pool

def conv_relu(bottom, ks, nout, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    relu = L.ReLU(conv, relu_param=dict(negative_slope=0.0), in_place=AF_INPLACE)
    return relu

def block(bottom, ks, nout, dilation=1, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, dilation=dilation, bias_term=BIAS_TERM, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
    relu = L.ReLU(conv, relu_param=dict(negative_slope=0.1), in_place=AF_INPLACE)
    ccat = L.Concat(relu, bottom, axis=1)
    relu = L.ReLU(ccat, relu_param=dict(negative_slope=0.0), in_place=AF_INPLACE)
    return relu

def fc_block(bottom, nout, drop):
    fc = L.InnerProduct(bottom,
                        num_output=nout,
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type="gaussian", std=0.005),
                        bias_filler=dict(type="constant", value=1))
    relu = L.ReLU(fc, in_place=AF_INPLACE)
    if drop != 0: 
        drop = L.Dropout(relu, dropout_ratio=0.5, in_place=False)
    return drop

def input_shifting(center, batch_size, grid_size):
    con = None
    for i in range(grid_size):
        if grid_size == 64:
            tx, ty = shift_value_8x8(i, SHIFT_VAL)
        elif grid_size == 25:
            i_pick = index_5x5_picker(i)
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

def image_data_center(batch_size=1, source='train_source', center_id=27):
    center, trash = L.ImageData(batch_size=batch_size,
                        source=DATA_PATH+'/'+source+str(center_id)+'.txt',
                        transform_param=dict(scale=1./256.),
                        shuffle=False,
                        ntop=2,
                        new_height=192,
                        new_width=192,
                        is_color=False)
    trash = L.Silence(trash, ntop=1)
    return center

def image_data(batch_size=1, source='train_source', grid_size=64):
    con = None
    trash = None
    trash_tot = None
    for i in range(grid_size):
        if grid_size == 64:
            i_pick = i
        elif grid_size == 25:
            i_pick = index_5x5_picker(i)
        else:
            print('This size is not supported')
        label, trash = L.ImageData(batch_size=batch_size,
                                source=DATA_PATH+'/'+source+str(i_pick)+'.txt',
                                transform_param=dict(scale=1./256.),
                                shuffle=False,
                                ntop=2,
                                new_height=192,
                                new_width=192,
                                is_color=False)
        if i == 0:
            con = label
        else:   
            con = L.Concat(*[con, label], concat_param={'axis': 1})
    #dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    #con_crop = L.Crop(con, dum_input, crop_param=dict(axis=2, offset=2))
        if i == 0:
            trash_tot = trash
        else:
            trash_tot = L.Eltwise(trash_tot, trash, operation=P.Eltwise.SUM)
    return con, trash_tot
    
def slice_warp(shift, flow_h, flow_v, slice_range):
    for i in range(slice_range):
        if i < slice_range-1:
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

def luma_layer(bottom):
    r, g, b = L.Slice(bottom, ntop=3, slice_param=dict(slice_dim=2, slice_point=[1,2]))
    y_r = L.Power(r, power=1.0, scale=0.299, shift=0.0, in_place=False)
    y_g = L.Power(g, power=1.0, scale=0.587, shift=0.0, in_place=False)
    y_b = L.Power(b, power=1.0, scale=0.114, shift=0.0, in_place=False)
    y = L.Eltwise(y_r, y_g, operation=P.Eltwise.SUM)
    y = L.Eltwise(y, y_b, operation=P.Eltwise.SUM)        
    return y

def axis_mean_loss_layer(bottom_predict, bottom_GT, res):
    slice_predict = None
    slice_GT = None
    remain_predict = bottom_predict
    remain_GT = bottom_GT
    mean_predict = None
    mean_GT = None
    loss = None
    loss_tot = None

    for i in range(res):
        if i < res-1:
            slice_predict, remain_predict = L.Slice(remain_predict, ntop=2, slice_param=dict(slice_dim=2, slice_point=[res]))
            slice_GT, remain_GT = L.Slice(remain_GT, ntop=2, slice_param=dict(slice_dim=2, slice_point=[res]))
        else:
            slice_predict = remain_predict
            slice_GT = remain_GT       

        slice_predict = L.Reduction(slice_predict, axis=2, operation=P.Reduction.MEAN)
        slice_GT = L.Reduction(slice_GT, axis=2, operation=P.Reduction.MEAN)
        
        loss = L.AbsLoss(slice_predict, slice_GT)
        if i == 0:
            loss_tot = loss
        else:
            loss_tot = L.Eltwise(loss_tot, loss, operation=P.Eltwise.SUM)

    slice_predict = None
    slice_GT = None
    remain_predict = bottom_predict
    remain_GT = bottom_GT
    mean_predict = None
    mean_GT = None
    loss = None
    #loss_tot = None

    return loss_tot

def axis_order_change_layer(bottom, batch_size, res):
    dum = L.DummyData(shape=dict(dim=[batch_size, 1, 188, 188]))
    crop = None
    concat = None

    for ax in range(res):
        for ay in range(res):
            crop = L.Crop(bottom, dum, crop_param=dict(axis=1, offset=[res*ay+ax, 0, 0]))
            if ax == 0 and ay == 0:
                concat = crop
            else:
                concat = L.Concat(*[concat, crop], concat_param={'axis': 1})

    crop = None
    #concat = None
    
    return concat

def denseUNet_train(batch_size=1):
    n = caffe.NetSpec()

    # Data loading
    n.input = image_data_center(batch_size, 'train_source', 27)

    # Network
    #n.dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    #n.input_crop = L.Crop(n.input, n.dum_input, crop_param=dict(axis=2, offset=2))   
    n.conv1, n.poo1 = conv_conv_downsample_layer(n.input, 3, 16, 2, 1)
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

    #n.flow64 = flow_layer(n.conv10, 64*2)
    n.flow25 = flow_layer(n.conv10, 25*2)


    n.flow25_mean = L.Reduction(n.flow25, axis=1, operation=P.Reduction.MEAN)
    n.flow25_mean_GT = L.DummyData(shape=dict(dim=[batch_size, 1]))
    n.flow25_mean_GT = L.Power(n.flow25_mean_GT, power=1.0, scale=0.0, shift=0.0, in_place=True)
    n.flow25_mean_loss = L.AbsLoss(n.flow25_mean, n.flow25_mean_GT)
    n.flow25_mean_loss = L.Power(n.flow25_mean_loss, power=1.0, scale=1000000.0, shift=0.0, in_place=True, loss_weight=1) #1000000

    #n.flow25_mean = L.Reduction(n.flow25_mean, axis=1, operation=P.Reduction.MEAN)
    #n.flow25_mean = L.Reduction(n.flow25_mean, axis=1, operation=P.Reduction.MEAN)
    
    # Translation
    #n.shift = input_shifting_8x8(n.input, batch_size)
    #n.flow_h, n.flow_v = L.Slice(n.flow64, ntop=2, slice_param=dict(slice_dim=1, slice_point=[64]))
    #n.predict = slice_warp(n.shift, n.flow_h, n.flow_v, 64)
    n.shift2 = input_shifting(n.input, batch_size, 25)
    n.flow_h2, n.flow_v2 = L.Slice(n.flow25, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.predict2 = slice_warp(n.shift2, n.flow_h2, n.flow_v2, 25)

    # Loss
    #n.label = image_data_8x8(batch_size, 'train_source')
    n.label2, n.trash_tot = image_data(batch_size, 'train_source', 25)
    n.dum_predict2 = L.DummyData(shape=dict(dim=[batch_size, 1, 188, 188]))
    n.predict2_crop = L.Crop(n.predict2, n.dum_predict2, crop_param=dict(axis=2, offset=2))  
    n.dum_label2 = L.DummyData(shape=dict(dim=[batch_size, 1, 188, 188]))
    n.label2_crop = L.Crop(n.label2, n.dum_label2, crop_param=dict(axis=2, offset=2))

    '''
    n.loss_ver = axis_mean_loss_layer(n.predict2_crop, n.label2_crop, 5)
    n.predict3_crop = axis_order_change_layer(n.predict2_crop, batch_size, 5)
    n.label3_crop = axis_order_change_layer(n.label2_crop, batch_size, 5)
    n.loss_hor = axis_mean_loss_layer(n.predict3_crop, n.label3_crop, 5)

    n.loss_tot = L.Eltwise(n.loss_ver, n.loss_hor, operation=P.Eltwise.SUM)
    n.loss_tot = L.Power(n.loss_tot, power=1.0, scale=5000.0, shift=0.0, in_place=True)
    '''
    #n.loss1 = L.AbsLoss(n.predict, n.label)
    n.loss2 = L.AbsLoss(n.predict2_crop, n.label2_crop)
    #n.loss1 = L.Power(n.loss1, power=1.0, scale=1.0, shift=0.0, in_place=True, loss_weight=1)
    n.loss2 = L.Power(n.loss2, power=1.0, scale=1.0, shift=0.0, in_place=True, loss_weight=1)


    n.predict2_edge = L.Python(n.predict2_crop, module = 'edge_layer', layer = 'EdgeLayer')
    n.label2_edge = L.Python(n.label2_crop, module = 'edge_layer', layer = 'EdgeLayer')
    n.loss3 = L.AbsLoss(n.predict2_edge, n.label2_edge)
    n.loss3 = L.Power(n.loss3, power=1.0, scale=1000.0, shift=0.0, in_place=True, loss_weight=1) # 1000


    return n.to_proto()

def denseUNet_test(batch_size=1):
    n = caffe.NetSpec()

    # Data loading
    n.input = image_data_center(batch_size, 'train_source', 27)
    
    # Network
    #n.dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    #n.input_crop = L.Crop(n.input, n.dum_input, crop_param=dict(axis=2, offset=2))   
    n.conv1, n.poo1 = conv_conv_downsample_layer(n.input, 3, 16, 2, 1)
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

    #n.flow64 = flow_layer(n.conv10, 64*2)
    n.flow25 = flow_layer(n.conv10, 25*2)

    # Translation
    # n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    # n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    # n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer', layer = 'BilinearSamplerLayer', ntop = 1)
    n.shift = input_shifting(n.input, batch_size, 25)
    n.flow_h, n.flow_v = L.Slice(n.flow25, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.predict = slice_warp(n.shift, n.flow_h, n.flow_v, 25)

    # Visualization
    n.label, n.trash_tot = image_data(batch_size, 'train_source', 25)
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

def denseUNet_solver(train_net_path=None, test_net_path=None, snapshot_path=None):
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
    s.base_lr = 0.0005 # 0.0005,  0.0000001 # 0.000005(basic), 0.0000001
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
    DATA_PATH = './datas/flower_dataset'
    MODEL_PATH = './models'
    TRAIN_PATH = './scripts/denseUNet_train.prototxt'
    TEST_PATH = './scripts/denseUNet_test.prototxt'
    SOLVER_PATH = './scripts/denseUNet_solver.prototxt'

    SHIFT_VAL = 0.8
    AF_INPLACE = True
    BIAS_TERM = True
    
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
    solver.net.copy_from('./models/denseUNet_solver_iter_17000.caffemodel')
    solver.solve()