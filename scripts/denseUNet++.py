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
    conv = L.Convolution(bottom, num_output=nout, kernel_size=ks, stride=strd, pad=pad, dilation=dil,
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

def upsample_concat_layer(bottom1=None, bottom2=None, nout=1, ks=3, strd=2, pad=0, batch_size=1, crop_size=0):
    deconv = L.Deconvolution(bottom1, convolution_param=dict(num_output=nout, kernel_size=ks, stride=strd, pad=pad))
    deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=AF_INPLACE)
    dum = L.DummyData(shape=dict(dim=[batch_size, nout, crop_size, crop_size]))
    deconv_crop = L.Crop(deconv, dum, crop_param=dict(axis=2, offset=0))
    conc = L.Concat(*[deconv_crop, bottom2], concat_param={'axis': 1})
    return conc

def image_data_center(batch_size=1, source='train_source', center_id=27):
    center, trash = L.ImageData(batch_size=batch_size,
                        source=DATA_PATH+'/'+source+str(center_id)+'.txt',
                        transform_param=dict(scale=1./256.),
                        shuffle=False,
                        ntop=2,
                        new_height=192,
                        new_width=192,
                        is_color=False)
    return center, trash

def image_data(batch_size=1, source='train_source', grid_size=64):
    for i in range(grid_size):
        if grid_size == 64:
            i_pick = i
        elif grid_size == 25:
            i_pick = index_picker_8x8(i)
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
            tot_trash = trash
        else:   
            con = L.Concat(*[con, label], concat_param={'axis': 1})
            tot_trash = L.Eltwise(tot_trash, trash, operation=P.Eltwise.SUM)
    #dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    #con_crop = L.Crop(con, dum_input, crop_param=dict(axis=2, offset=2))
    return con, tot_trash

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

def denseUNet_train(batch_size=1):
    n = caffe.NetSpec()

    # Data loading
    n.input, n.trash = image_data_center(batch_size, 'train_source', 27)

    # Network
    #n.dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    #n.input_crop = L.Crop(n.input, n.dum_input, crop_param=dict(axis=2, offset=2))   
    n.conv1, n.poo1 = conv_conv_downsample_layer(n.input, 16, 3, 2, 1)
    n.conv2, n.poo2 = conv_conv_downsample_layer(n.poo1, 32, 3, 2, 1)
    n.conv3, n.poo3 = conv_conv_downsample_layer(n.poo2, 64, 3, 2, 1)
    n.conv4, n.poo4 = conv_conv_downsample_layer(n.poo3, 128, 3, 2, 1)
    n.conv5, n.poo5 = conv_conv_downsample_layer(n.poo4, 256, 3, 2, 1)
    n.feature = conv_conv_res_dense_layer(n.poo5, 512, 3, 1, 1, 1)

    n.deconv_in_1_1 = upsample_concat_layer(n.conv2, n.conv1, 64, 3, 2, 0, batch_size, 192)
    n.conv_in_1_1 = conv_conv_layer(n.deconv_in_1_1, 64, 3, 1, 1, 1)



    n.deconv_in_2_1 = upsample_concat_layer(n.conv3, n.conv2, 64, 3, 2, 0, batch_size, 96)
    n.conv_in_2_1 = conv_conv_layer(n.deconv_in_2_1, 64, 3, 1, 1, 1)
    n.deconv_in_2_2 = upsample_concat_layer(n.conv_in_2_1, n.conv_in_1_1, 64, 3, 2, 0, batch_size, 192)
    n.conv_in_2_2 = conv_conv_layer(n.deconv_in_2_2, 64, 3, 1, 1, 1)



    n.deconv_in_3_1 = upsample_concat_layer(n.conv4, n.conv3, 64, 3, 2, 0, batch_size, 48)
    n.conv_in_3_1 = conv_conv_layer(n.deconv_in_3_1, 64, 3, 1, 1, 1)
    n.deconv_in_3_2 = upsample_concat_layer(n.conv_in_3_1, n.conv_in_2_1, 64, 3, 2, 0, batch_size, 96)
    n.conv_in_3_2 = conv_conv_layer(n.deconv_in_3_2, 64, 3, 1, 1, 1)
    n.deconv_in_3_3 = upsample_concat_layer(n.conv_in_3_2, n.conv_in_2_2, 64, 3, 2, 0, batch_size, 192)
    n.conv_in_3_3 = conv_conv_layer(n.deconv_in_3_3, 64, 3, 1, 1, 1)



    n.deconv_in_4_1 = upsample_concat_layer(n.conv5, n.conv4, 128, 3, 2, 0, batch_size, 24)
    n.conv_in_4_1 = conv_conv_layer(n.deconv_in_4_1, 128, 3, 1, 1, 1)
    n.deconv_in_4_2 = upsample_concat_layer(n.conv_in_4_1, n.conv_in_3_1, 64, 3, 2, 0, batch_size, 48)
    n.conv_in_4_2 = conv_conv_layer(n.deconv_in_4_2, 64, 3, 1, 1, 1)
    n.deconv_in_4_3 = upsample_concat_layer(n.conv_in_4_2, n.conv_in_3_2, 64, 3, 2, 0, batch_size, 96)
    n.conv_in_4_3 = conv_conv_layer(n.deconv_in_4_3, 64, 3, 1, 1, 1)
    n.deconv_in_4_4 = upsample_concat_layer(n.conv_in_4_3, n.conv_in_3_3, 64, 3, 2, 0, batch_size, 192)
    n.conv_in_4_4 = conv_conv_layer(n.deconv_in_4_4, 64, 3, 1, 1, 1)






    n.deconv1 = upsample_concat_layer(n.feature, n.conv5, 256, 3, 2, 0, batch_size, 12)
    n.conv6 = conv_conv_layer(n.deconv1, 256, 3, 1, 1, 1)
    n.deconv2 = upsample_concat_layer(n.conv6, n.conv4, 128, 3, 2, 0, batch_size, 24)
    n.conv7 = conv_conv_layer(n.deconv2, 128, 3, 1, 1, 1)
    n.deconv3 = upsample_concat_layer(n.conv7, n.conv3, 64, 3, 2, 0, batch_size, 48)
    n.conv8 = conv_conv_layer(n.deconv3, 64, 3, 1, 1, 1)
    n.deconv4 = upsample_concat_layer(n.conv8, n.conv2, 64, 3, 2, 0, batch_size, 96)
    n.conv9 = conv_conv_layer(n.deconv4, 64, 3, 1, 1, 1)
    n.deconv5 = upsample_concat_layer(n.conv9, n.conv1, 64, 3, 2, 0, batch_size, 192)
    n.conv10 = conv_conv_layer(n.deconv5, 64, 3, 1, 1, 1)




    n.conc_tot = L.Concat(*[n.conv10, n.conv_in_4_4, n.conv_in_3_3, n.conv_in_2_2, n.conv_in_1_1], concat_param={'axis': 1})





    n.flow25 = flow_layer(n.conc_tot, 25*2)

    '''
    n.flow25_mean = L.Reduction(n.flow25, axis=1, operation=P.Reduction.MEAN)
    n.flow25_mean_GT = L.DummyData(shape=dict(dim=[batch_size, 1]))
    n.flow25_mean_GT = L.Power(n.flow25_mean_GT, power=1.0, scale=0.0, shift=0.0, in_place=True)
    n.flow25_mean_loss = L.AbsLoss(n.flow25_mean, n.flow25_mean_GT)
    n.flow25_mean_loss = L.Power(n.flow25_mean_loss, power=1.0, scale=1000.0, shift=0.0, in_place=True, loss_weight=1) #1000000
    '''
    
    # Translation
    n.shift = input_shifting(n.input, batch_size, 25)
    n.flow_h, n.flow_v = L.Slice(n.flow25, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.predict = slice_warp(n.shift, n.flow_h, n.flow_v, 25)

    # Loss
    n.label, n.trash_tot = image_data(batch_size, 'train_source', 25)
    n.dum_predict = L.DummyData(shape=dict(dim=[batch_size, 1, 188, 188]))
    n.predict_crop = L.Crop(n.predict, n.dum_predict, crop_param=dict(axis=2, offset=2))  
    n.dum_label = L.DummyData(shape=dict(dim=[batch_size, 1, 188, 188]))
    n.label_crop = L.Crop(n.label, n.dum_label, crop_param=dict(axis=2, offset=2))

    n.loss = L.AbsLoss(n.predict_crop, n.label_crop)
    #n.loss = L.EuclideanLoss(n.predict_crop, n.label_crop)
    n.loss = L.Power(n.loss, power=1.0, scale=1.0, shift=0.0, in_place=True, loss_weight=1)

    return n.to_proto()

def denseUNet_test(batch_size=1):
    n = caffe.NetSpec()

    # Data loading
    n.input, n.trash = image_data_center(batch_size, 'train_source', 27)

    # Network
    #n.dum_input = L.DummyData(shape=dict(dim=[batch_size, 1, 192, 192]))
    #n.input_crop = L.Crop(n.input, n.dum_input, crop_param=dict(axis=2, offset=2))   
    n.conv1, n.poo1 = conv_conv_downsample_layer(n.input, 16, 3, 2, 1)
    n.conv2, n.poo2 = conv_conv_downsample_layer(n.poo1, 32, 3, 2, 1)
    n.conv3, n.poo3 = conv_conv_downsample_layer(n.poo2, 64, 3, 2, 1)
    n.conv4, n.poo4 = conv_conv_downsample_layer(n.poo3, 128, 3, 2, 1)
    n.conv5, n.poo5 = conv_conv_downsample_layer(n.poo4, 256, 3, 2, 1)
    n.feature = conv_conv_res_dense_layer(n.poo5, 512, 3, 1, 1, 1)

    n.deconv_in_1_1 = upsample_concat_layer(n.conv2, n.conv1, 64, 3, 2, 0, batch_size, 192)
    n.conv_in_1_1 = conv_conv_layer(n.deconv_in_1_1, 64, 3, 1, 1, 1)



    n.deconv_in_2_1 = upsample_concat_layer(n.conv3, n.conv2, 64, 3, 2, 0, batch_size, 96)
    n.conv_in_2_1 = conv_conv_layer(n.deconv_in_2_1, 64, 3, 1, 1, 1)
    n.deconv_in_2_2 = upsample_concat_layer(n.conv_in_2_1, n.conv_in_1_1, 64, 3, 2, 0, batch_size, 192)
    n.conv_in_2_2 = conv_conv_layer(n.deconv_in_2_2, 64, 3, 1, 1, 1)



    n.deconv_in_3_1 = upsample_concat_layer(n.conv4, n.conv3, 64, 3, 2, 0, batch_size, 48)
    n.conv_in_3_1 = conv_conv_layer(n.deconv_in_3_1, 64, 3, 1, 1, 1)
    n.deconv_in_3_2 = upsample_concat_layer(n.conv_in_3_1, n.conv_in_2_1, 64, 3, 2, 0, batch_size, 96)
    n.conv_in_3_2 = conv_conv_layer(n.deconv_in_3_2, 64, 3, 1, 1, 1)
    n.deconv_in_3_3 = upsample_concat_layer(n.conv_in_3_2, n.conv_in_2_2, 64, 3, 2, 0, batch_size, 192)
    n.conv_in_3_3 = conv_conv_layer(n.deconv_in_3_3, 64, 3, 1, 1, 1)



    n.deconv_in_4_1 = upsample_concat_layer(n.conv5, n.conv4, 128, 3, 2, 0, batch_size, 24)
    n.conv_in_4_1 = conv_conv_layer(n.deconv_in_4_1, 128, 3, 1, 1, 1)
    n.deconv_in_4_2 = upsample_concat_layer(n.conv_in_4_1, n.conv_in_3_1, 64, 3, 2, 0, batch_size, 48)
    n.conv_in_4_2 = conv_conv_layer(n.deconv_in_4_2, 64, 3, 1, 1, 1)
    n.deconv_in_4_3 = upsample_concat_layer(n.conv_in_4_2, n.conv_in_3_2, 64, 3, 2, 0, batch_size, 96)
    n.conv_in_4_3 = conv_conv_layer(n.deconv_in_4_3, 64, 3, 1, 1, 1)
    n.deconv_in_4_4 = upsample_concat_layer(n.conv_in_4_3, n.conv_in_3_3, 64, 3, 2, 0, batch_size, 192)
    n.conv_in_4_4 = conv_conv_layer(n.deconv_in_4_4, 64, 3, 1, 1, 1)






    n.deconv1 = upsample_concat_layer(n.feature, n.conv5, 256, 3, 2, 0, batch_size, 12)
    n.conv6 = conv_conv_layer(n.deconv1, 256, 3, 1, 1, 1)
    n.deconv2 = upsample_concat_layer(n.conv6, n.conv4, 128, 3, 2, 0, batch_size, 24)
    n.conv7 = conv_conv_layer(n.deconv2, 128, 3, 1, 1, 1)
    n.deconv3 = upsample_concat_layer(n.conv7, n.conv3, 64, 3, 2, 0, batch_size, 48)
    n.conv8 = conv_conv_layer(n.deconv3, 64, 3, 1, 1, 1)
    n.deconv4 = upsample_concat_layer(n.conv8, n.conv2, 64, 3, 2, 0, batch_size, 96)
    n.conv9 = conv_conv_layer(n.deconv4, 64, 3, 1, 1, 1)
    n.deconv5 = upsample_concat_layer(n.conv9, n.conv1, 64, 3, 2, 0, batch_size, 192)
    n.conv10 = conv_conv_layer(n.deconv5, 64, 3, 1, 1, 1)




    n.conc_tot = L.Concat(*[n.conv10, n.conv_in_4_4, n.conv_in_3_3, n.conv_in_2_2, n.conv_in_1_1], concat_param={'axis': 1})





    n.flow25 = flow_layer(n.conc_tot, 25*2)

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
                    param_str=str(dict(path='./datas/flower_dataset', name='flow_h', mult=110)))
    n.trash2 = L.Python(n.flow_v, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/flower_dataset', name='flow_v', mult=110)))
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
    s.type = 'Adam'
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

    s.snapshot = 100

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
            f.write(str(denseUNet_train(4)))    
        with open(TEST_PATH, 'w') as f:
            f.write(str(denseUNet_test(1)))
    
    def generate_solver():
        with open(SOLVER_PATH, 'w') as f:
            f.write(str(denseUNet_solver(TRAIN_PATH, TEST_PATH, MODEL_PATH)))

    generate_net()
    generate_solver()
    solver = caffe.get_solver(SOLVER_PATH)
    #solver.net.copy_from('./models/denseUNet_solver_iter_2000.caffemodel')
    solver.solve()