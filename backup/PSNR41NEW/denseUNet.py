# +---+----+----+----+----+----+----+----+----+
# | 0 | 9  | 18 | 27 | 36 | 45 | 54 | 63 | 72 |
# +---+----+----+----+----+----+----+----+----+
# | 1 | 10 | 19 | 28 | 37 | 46 | 55 | 64 | 73 |
# +---+----+----+----+----+----+----+----+----+
# | 2 | 11 | 20 | 29 | 38 | 47 | 56 | 65 | 74 |
# +---+----+----+----+----+----+----+----+----+
# | 3 | 12 | 21 | 30 | 39 | 48 | 57 | 66 | 75 |
# +---+----+----+----+----+----+----+----+----+
# | 4 | 13 | 22 | 31 | 40 | 49 | 58 | 67 | 76 |
# +---+----+----+----+----+----+----+----+----+
# | 5 | 14 | 23 | 32 | 41 | 50 | 59 | 68 | 77 |
# +---+----+----+----+----+----+----+----+----+
# | 6 | 15 | 24 | 33 | 42 | 51 | 60 | 69 | 78 |
# +---+----+----+----+----+----+----+----+----+
# | 7 | 16 | 25 | 34 | 43 | 52 | 61 | 70 | 79 |
# +---+----+----+----+----+----+----+----+----+
# | 8 | 17 | 26 | 35 | 44 | 53 | 62 | 71 | 80 |
# +---+----+----+----+----+----+----+----+----+

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
import time
import math
import argparse
import lf_func

np.set_printoptions(threshold=sys.maxsize) # Numpy maximum print

caffe.set_mode_gpu() 

####################################################################################################
##########                               Public Function                                  ##########
####################################################################################################

def make_colorwheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def trans_SAIs_to_LF(imgSAIs):
    imgLF = np.zeros((256*1*5, 256*1*5, 3))
    full_LF_crop = np.zeros((256, 256, 3, 5, 5))
    for ax in range(5):
        for ay in range(5):
            full_LF_crop[:, :, :, ax, ay] = imgSAIs[:, :, :, 5*ax+ay]
    for ax in range(5):
        for ay in range(5):
            resized2 = full_LF_crop[:, :, :, ay, ax]
            resized2 = cv2.resize(resized2, dsize=(256*1, 256*1), interpolation=cv2.INTER_CUBIC) 
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

def shift_value_5x5(i, shift_val):
    if i<=4:
        tx = -2*shift_val
    elif i>4 and i<=9:
        tx = -1*shift_val
    elif i>9 and i<=14:
        tx = 0
    elif i>14 and i<=19:
        tx = 1*shift_val
    elif i>19 and i<=24:
        tx = 2*shift_val
    else:
        tx = 3*shift_val
    if i == 0 or (i)%5==0:
        ty = -2*shift_val
    elif i == 1 or (i-1)%5==0:
        ty = -1*shift_val
    elif i == 2 or (i-2)%5==0:
        ty = 0
    elif i == 3 or (i-3)%5==0:
        ty = 1*shift_val
    elif i == 4 or (i-4)%5==0:
        ty = 2*shift_val
    else:
        ty = 3*shift_val
    return tx, ty

def index_picker_5x5(i, pick_mode='9x9'):
    if pick_mode == '9x9':
        id_list = [20, 21, 22, 23, 24,
                    29, 30, 31, 32, 33,
                    38, 39, 40, 41, 42,
                    47, 48, 49, 50, 51,
                    56, 57, 58, 59, 60]
    elif pick_mode == '8x8':
        id_list = [9, 10, 11, 12, 13,
                    17, 18, 19, 20, 21,
                    25, 26, 27, 28, 29,
                    33, 34, 35, 36, 37,
                    41, 42, 43, 44, 45]
    return id_list[i]

def image_data_center(batch_size, data_path, center_id):
    center, trash = L.ImageData(batch_size=batch_size,
                        source=data_path+'/dataset_list'+str(center_id)+'.txt',
                        transform_param=dict(scale=1./256.),
                        shuffle=False,
                        ntop=2,
                        new_height=256,
                        new_width=256,
                        is_color=False)
    return center

def image_data_depth(batch_size, data_path):
    depth, trash = L.ImageData(batch_size=batch_size,
                        source=data_path+'/depthset_list.txt',
                        transform_param=dict(scale=1./256.),
                        shuffle=False,
                        ntop=2,
                        new_height=256,
                        new_width=256,
                        is_color=False)
    return depth

def image_data(batch_size, data_path, n_sai):
    for i_sai in range(n_sai):
        label, trash = L.ImageData(batch_size=batch_size,
                                source=data_path+'/dataset_list'+str(i_sai)+'.txt',
                                transform_param=dict(scale=1./256.),
                                shuffle=False,
                                ntop=2,
                                new_height=256,
                                new_width=256,
                                is_color=False)
        if i_sai == 0:
            con = label
        else:   
            con = L.Concat(*[con, label], concat_param={'axis': 1})
    return con

def input_shifting(src, n_sai, shift_val):
    for i_sai in range(25):
        tx, ty = shift_value_5x5(i_sai, shift_val)
        param_str = json.dumps({'tx': tx, 'ty': ty})
        shift = L.Python(src, module = 'input_shifting_layer', layer = 'InputShiftingLayer', ntop = 1, param_str = param_str)
        if i_sai == 0:
            con = shift
        else:   
            con = L.Concat(*[con, shift], concat_param={'axis': 1})
    return con

def slice_warp(src, flow_h, flow_v, n_sai):
    for i in range(n_sai):
        if i < n_sai-1:
            src_slice, src = L.Slice(src, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
            flow_h_slice, flow_h = L.Slice(flow_h, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
            flow_v_slice, flow_v = L.Slice(flow_v, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
            predict_slice = L.Warping(src_slice, flow_h_slice, flow_v_slice)
        else:
            src_slice = src
            flow_h_slice = flow_h
            flow_v_slice = flow_v
            predict_slice = L.Warping(src_slice, flow_h_slice, flow_v_slice)
        if i == 0:
            con = predict_slice
        else:
            con = L.Concat(*[con, predict_slice], concat_param={'axis': 1})
    return con

def slice_warp2(src, flow_h, flow_v, n_sai):
    for i in range(n_sai):
        if i < n_sai-1:
            src_slice, src = L.Slice(src, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
            flow_h_slice, flow_h = L.Slice(flow_h, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
            flow_v_slice, flow_v = L.Slice(flow_v, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
            flow_hv_slice = L.Concat(*[flow_h_slice, flow_v_slice], concat_param={'axis': 1})
            predict_slice = L.FlowWarp(src_slice, flow_hv_slice)
        else:
            src_slice = src
            flow_h_slice = flow_h
            flow_v_slice = flow_v
            flow_hv_slice = L.Concat(*[flow_h_slice, flow_v_slice], concat_param={'axis': 1})
            predict_slice = L.FlowWarp(src_slice, flow_hv_slice)
        if i == 0:
            con = predict_slice
        else:
            con = L.Concat(*[con, predict_slice], concat_param={'axis': 1})
    return con

def change_order(imgs, batch_size):
    imgs_result = None
    for j in range(5):
        for i in range(5):
            dum = L.DummyData(shape=dict(dim=[batch_size, 1, 256, 256]))
            temp = L.Crop(imgs, dum, crop_param=dict(axis=1, offset=[5*i+j,0,0]))

            if i==0 and j==0:
                imgs_result = temp
            else:
                imgs_result = L.Concat(*[imgs_result, temp], concat_param={'axis': 1})
    return imgs_result

def l_loss(imgs, lables, batch_size):
    loss = None
    loss2 = None

    imgs2 = change_order(imgs, batch_size)
    lables2 = change_order(lables, batch_size)
    
    for i in range(5):
        if i < 4:
            imgs_slice, imgs = L.Slice(imgs, ntop=2, slice_param=dict(slice_dim=1, slice_point=[5]))
            imgs2_slice, imgs2 = L.Slice(imgs2, ntop=2, slice_param=dict(slice_dim=1, slice_point=[5]))
            lables_slice, lables = L.Slice(lables, ntop=2, slice_param=dict(slice_dim=1, slice_point=[5]))
            lables2_slice, lables2 = L.Slice(lables2, ntop=2, slice_param=dict(slice_dim=1, slice_point=[5]))
        else:
            imgs_slice = imgs
            imgs2_slice = imgs2
            lables_slice = lables
            lables2_slice = lables2

        imgs_redu = L.Reduction(imgs_slice, axis=1, operation=P.Reduction.MEAN)
        imgs2_redu = L.Reduction(imgs2_slice, axis=1, operation=P.Reduction.MEAN)
        lables_redu = L.Reduction(lables_slice, axis=1, operation=P.Reduction.MEAN)
        lables2_redu = L.Reduction(lables2_slice, axis=1, operation=P.Reduction.MEAN)
        
        if i == 0:
            loss = L.AbsLoss(imgs_redu, lables_redu)
            loss2 = L.AbsLoss(imgs2_redu, lables2_redu)
        else:
            temp_loss = L.AbsLoss(imgs_redu, lables_redu)
            temp_loss2 = L.AbsLoss(imgs2_redu, lables2_redu)
            loss = L.Eltwise(loss, temp_loss, operation=P.Eltwise.SUM)
            loss2 = L.Eltwise(loss2, temp_loss2, operation=P.Eltwise.SUM)

    return loss, loss2

def flat(imgs, batch_size, ch=3):
    imgs_vector = None
    imgs_matrix = None
    for j in range(5):
        for i in range(5):
            dum = L.DummyData(shape=dict(dim=[batch_size, ch, 256, 256]))
            temp = L.Crop(imgs, dum, crop_param=dict(axis=1, offset=[ch*(5*i+j),0,0]))

            if i==0:
                imgs_vector = temp
            else:
                imgs_vector = L.Concat(*[imgs_vector, temp], concat_param={'axis': 3})
        if j==0:
            imgs_matrix = imgs_vector
        else:
            imgs_matrix = L.Concat(*[imgs_matrix, imgs_vector], concat_param={'axis': 2})
    return imgs_matrix

####################################################################################################
##########                             Fundamental Network                                ##########
####################################################################################################

def denseUNet(input, batch_size):
    def flow_layer(bottom=None, nout=1):
        flow_init = L.Convolution(bottom, num_output=1, kernel_size=3, stride=1, dilation=1, pad=1,
                                group=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        #flow_init = L.ReLU(flow_init, in_place=True, engine=1)
        flow = L.Convolution(flow_init, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                group=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        return flow, flow_init

    def conv_conv_group_layer(bottom=None, nout=1):
        conv = L.Convolution(bottom, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True, engine=1)
        conv = L.Convolution(conv, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                group=nout, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        conv = L.Convolution(conv, num_output=nout, kernel_size=1, stride=1, dilation=1, pad=0,
                                bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True, engine=1)
        return conv

    def conv_conv_res_dense_layer(bottom=None, nout=1):
        bottom = L.Convolution(bottom, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        bottom = L.ReLU(bottom, relu_param=dict(negative_slope=0.2), in_place=True, engine=1)
        conv = L.Convolution(bottom, num_output=nout, kernel_size=3, stride=1, dilation=3, pad=3,
                                bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True, engine=1)
        conv = L.Convolution(conv, num_output=nout, kernel_size=3, stride=1, dilation=6, pad=6,
                                bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        elth = L.Eltwise(conv, bottom, operation=P.Eltwise.SUM)
        elth = L.ReLU(elth, relu_param=dict(negative_slope=0.2), in_place=True, engine=1)
        return elth   

    def conv_conv_downsample_group_layer(bottom=None, nout=1):
        conv = L.Convolution(bottom, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True, engine=1)
        conv = L.Convolution(conv, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                group=nout, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        conv = L.Convolution(conv, num_output=nout, kernel_size=1, stride=1, dilation=1, pad=0,
                                bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True, engine=1)
        pool = L.Pooling(conv, kernel_size=3, stride=2, pool=P.Pooling.MAX)
        pool = L.ReLU(pool, relu_param=dict(negative_slope=0.2), in_place=True, engine=1)
        return conv, pool

    def upsample_concat_layer(bottom1=None, bottom2=None, nout=1, crop_size=0):
        deconv = L.Deconvolution(bottom1, convolution_param=dict(num_output=nout, kernel_size=4, stride=2, pad=1))
        deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=True)
        #dum = L.DummyData(shape=dict(dim=[batch_size, nout, crop_size, crop_size]))
        #deconv_crop = L.Crop(deconv, dum, crop_param=dict(axis=1, offset=[0, 0, 0]))
        conc = L.Concat(*[deconv, bottom2], concat_param={'axis': 1})
        return conc

    conv1, poo1 = conv_conv_downsample_group_layer(input, 16)
    conv2, poo2 = conv_conv_downsample_group_layer(poo1, 32)
    conv3, poo3 = conv_conv_downsample_group_layer(poo2, 64)
    conv4, poo4 = conv_conv_downsample_group_layer(poo3, 128)
    conv5, poo5 = conv_conv_downsample_group_layer(poo4, 256)
    
    feature = conv_conv_res_dense_layer(poo5, 256)

    deconv1 = upsample_concat_layer(feature, conv5, 256, 16)
    deconv1 = conv_conv_group_layer(deconv1, 256)
    deconv2 = upsample_concat_layer(deconv1, conv4, 128, 32)
    deconv2 = conv_conv_group_layer(deconv2, 128)
    deconv3 = upsample_concat_layer(deconv2, conv3, 64, 64)
    deconv3 = conv_conv_group_layer(deconv3, 64)
    deconv4 = upsample_concat_layer(deconv3, conv2, 32, 128)
    deconv4 = conv_conv_group_layer(deconv4, 32)
    deconv5 = upsample_concat_layer(deconv4, conv1, 16, 256)
    deconv5 = conv_conv_group_layer(deconv5, 16)

    flow, flow_init = flow_layer(deconv5, 25*2)

    return flow, flow_init

####################################################################################################
##########                               Network Generator                                ##########
####################################################################################################

'''
    f = open(args.trainset_path+'/depthset_list.txt', 'w')
    for i_tot in range(args.train_size):
        data = args.trainset_path+'_depth/sai'+str(i_tot)+'_'+str(40)+'.png 0'+'\n'
        f.write(data)
    f.close()
'''

def denseUNet_train(args):
    # Make data list
    for i_sai in range(args.n_sai):
        i_pick = index_picker_5x5(i_sai, args.pick_mode)
        f = open(args.trainset_path+'/dataset_list'+str(i_sai)+'.txt', 'w')
        for i_tot in range(args.train_size):
            data = args.trainset_path+'/sai'+str(i_tot)+'_'+str(i_pick)+'.png 0'+'\n'
            f.write(data)
        f.close()

    # Init network   
    n = caffe.NetSpec()

    # Input
    n.input = image_data_center(batch_size=args.batch_size, data_path=args.trainset_path, center_id=args.center_id)
    n.label = image_data(batch_size=args.batch_size, data_path=args.trainset_path, n_sai=args.n_sai)

    # Fundamental
    n.flow, n.flow_init = denseUNet(n.input, args.batch_size)

    # Translation
    n.shift = input_shifting(src=n.input, n_sai=args.n_sai, shift_val=args.shift_val)
    n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[args.n_sai]))
    n.predict = slice_warp2(n.shift, n.flow_h, n.flow_v, args.n_sai)

    # Loss   
    n.lossA = L.AbsLoss(n.predict, n.label)
    n.lossA = L.Power(n.lossA, power=1.0, scale=1.0, shift=0.0, loss_weight=1)    

    # Generate Prototxt
    with open(args.train_path, 'w') as f:
        f.write(str(n.to_proto()))    

def denseUNet_test(args):
    # Generate dataset list
    for i_sai in range(args.n_sai):
        i_pick = index_picker_5x5(i_sai, args.pick_mode)
        f = open(args.testset_path+'/dataset_list'+str(i_sai)+'.txt', 'w')
        for i_tot in range(args.test_size):
            data = args.testset_path+'/sai'+str(i_tot)+'_'+str(i_pick)+'.png 0'+'\n'
            f.write(data)
        f.close()

    # Init network       
    n = caffe.NetSpec()

    # Input
    n.input = image_data_center(batch_size=args.batch_size, data_path=args.testset_path, center_id=args.center_id)
    n.label = image_data(batch_size=args.batch_size, data_path=args.testset_path, n_sai=args.n_sai)

    # Fundamental
    n.flow, n.flow_init = denseUNet(n.input, args.batch_size)   

    # Translation
    n.shift = input_shifting(src=n.input, n_sai=args.n_sai, shift_val=args.shift_val)
    n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[args.n_sai]))
    n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    param_str = json.dumps({'flow_size': args.n_sai, 'color_size': 1})
    n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1, param_str = param_str)

    # Visualization
    n.trash1 = L.Python(n.flow_h, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset', name='flow_h', mult=20)))
    n.silence1 = L.Silence(n.trash1, ntop=0)
    n.trash2 = L.Python(n.flow_v, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset', name='flow_v', mult=20)))
    n.silence2 = L.Silence(n.trash2, ntop=0)
    n.trash3 = L.Python(n.shift, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset', name='input', mult=1*256)))
    n.silence3 = L.Silence(n.trash3, ntop=0)
    n.trash4 = L.Python(n.label, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset', name='label', mult=1*256)))
    n.silence4 = L.Silence(n.trash4, ntop=0)
    n.trash5 = L.Python(n.predict, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset', name='predict', mult=1*256)))
    n.silence5 = L.Silence(n.trash5, ntop=0)
    n.trash6 = L.Python(n.flow_init, module='print_scaled', layer='PrintScaledLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset/depth_predict.png')))
    n.silence6 = L.Silence(n.trash6, ntop=0)

    # Generate Prototxt
    with open(args.test_path, 'w') as f:
        f.write(str(n.to_proto()))

def denseUNet_deploy(args):
    # Init network  
    n = caffe.NetSpec()

    # Input
    n.input = L.Input(input_param=dict(shape=dict(dim=[1, 3, 256, 256])))
    n.input_luma = L.Input(input_param=dict(shape=dict(dim=[1, 1, 256, 256])))
    n.input_luma = L.Power(n.input_luma, power=1.0, scale=1./256., shift=0, in_place=True)

    # Fundamental
    n.flow, n.flow_init = denseUNet(n.input_luma, batch_size)
    n.silence = L.Silence(n.flow_init, ntop=0)  

    # Translation
    n.shift = input_shifting(center=n.input, batch_size=batch_size, sai=sai, shift_val=shift_val)
    n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    param_str = json.dumps({'flow_size': 25, 'color_size': 3})
    n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1, param_str = param_str)

    # Generate Prototxt
    with open(script_path, 'w') as f:
        f.write(str(n.to_proto()))

def denseUNet_deploy2(args):
    # Init network  
    n = caffe.NetSpec()

    # Input
    if args.mode == 'deploy':
        n.input, n.trash = L.ImageData(batch_size=1,
                            source='./input_list.txt',
                            transform_param=dict(scale=1./256.),
                            shuffle=False,
                            ntop=2,
                            new_height=256,
                            new_width=256,
                            is_color=True)
        n.input_luma, n.trash = L.ImageData(batch_size=1,
                            source='./input_list.txt',
                            transform_param=dict(scale=1./256.),
                            shuffle=False,
                            ntop=2,
                            new_height=256,
                            new_width=256,
                            is_color=False)
    elif args.mode == 'test':
        n.input = L.Input(input_param=dict(shape=dict(dim=[1, 3, 256, 256])))
        n.input_luma = L.Input(input_param=dict(shape=dict(dim=[1, 1, 256, 256])))
        n.input_luma_down = L.Power(n.input_luma, power=1.0, scale=1./256., shift=0, in_place=False)

    # Fundamental
    n.flow, n.flow_init = denseUNet(n.input_luma_down, batch_size=1)
    n.silence = L.Silence(n.flow_init, ntop=0)  

    # Translation
    n.shift = input_shifting(src=n.input, n_sai=args.n_sai, shift_val=args.shift_val)
    n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[args.n_sai]))
    n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    param_str = json.dumps({'flow_size': args.n_sai, 'color_size': 3})
    n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1, param_str = param_str)

    # Visualization
    n.result = flat(n.predict, batch_size=1, ch=3)

    n.trash1 = L.Python(n.result, module='print_layer', layer='PrintLayer', ntop=1,
                    param_str=str(dict(path='./', name='flow_result', mult=1)))
    n.silence1 = L.Silence(n.trash1, ntop=0)
    n.trash2 = L.Python(n.flow_init, module='print_scaled', layer='PrintScaledLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset/depth_predict.png')))
    n.silence2 = L.Silence(n.trash2, ntop=0)

    # Generate Prototxt
    with open('./deploy.prototxt', 'w') as f:
        f.write(str(n.to_proto()))

####################################################################################################
##########                                Solver Generator                                ##########
####################################################################################################
    
def denseUNet_solver(args):
    s = caffe_pb2.SolverParameter()

    s.train_net = args.train_path

    if args.test_path is not None:
        s.test_net.append(args.test_path)
        s.test_interval = 20
        s.test_iter.append(1)
    else:
        s.test_initialization = False
    
    s.iter_size = 1
    s.max_iter = args.epoch*args.train_size
    s.type = 'Adam'
    s.base_lr = args.lr # 0.0005,  0.0000001 # 0.000005(basic), 0.0000001
    s.lr_policy = 'step'
    s.gamma = 0.75
    s.power = 0.75
    s.stepsize = 1000
    s.momentum = 0.999
    s.momentum2 = 0.999
    s.weight_decay = 0.0005
    s.clip_gradients = 1
    s.display = 1

    s.snapshot = 1000

    if args.model_path is not None:
        s.snapshot_prefix = args.model_path

    s.solver_mode = caffe_pb2.SolverParameter.GPU

    with open(args.solver_path, 'w') as f:
        f.write(str(s))

####################################################################################################
##########                                    Tester                                      ##########
####################################################################################################

def denseUNet_tester(args):
    import skimage

    # Params
    sai_w = 256
    sai_h = 256
    ang_w = 5
    ang_h = 5
    sai_amount = 25
    sai_i_center = math.floor(math.sqrt(sai_amount))
    pr_path = './output/predict'
    gt_path = './output/GT'
    pick_mode = '9x9'
    test_amount = 98

    # Set model
    n = caffe.Net('./deploy.prototxt', './models/denseUNet_iter_10000.caffemodel', caffe.TEST)

    psnr_mean = 0
    ssim_mean = 0
    time_mean = 0
    for i_tot in range(test_amount):
        start = time.time()

        # Load image
        sai_i_center2 = index_picker_5x5(sai_i_center, pick_mode)
        src_img = cv2.imread(args.testset_path+'/sai'+str(i_tot)+'_'+str(40)+'.png', 1)
        src_img = cv2.resize(src_img, (sai_w, sai_h), interpolation=cv2.INTER_AREA)
        luma_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

        # Input image
        src_blob = np.zeros((1, 3, sai_h, sai_w))
        luma_blob = np.zeros((1, 1, sai_h, sai_w))
        for i in range(3):
            src_blob[0, i, :, :] = src_img[:, :, i]
        luma_blob[0, 0, :, :] = luma_img[:, :]
        n.blobs['input'].data[...] = src_blob
        n.blobs['input_luma'].data[...] = luma_blob

        # Run model
        n.forward()

        ### Print predict ####
        # Get SAI, flow list
        sai_img = np.zeros((sai_h, sai_w, 3))
        flow_img = np.zeros((sai_h, sai_w, 2))
        sai_img_list =  np.zeros((sai_h, sai_w, 3, sai_amount))
        flow_img_list = np.zeros((sai_h, sai_w, 3, sai_amount))
        dst_blob = n.blobs['predict'].data[...]
        flow_v_blob = n.blobs['flow_v'].data[...]
        flow_h_blob = n.blobs['flow_h'].data[...]
        for i in range(sai_amount):
            dst_blob_slice = dst_blob[:, 3*i:3*i+3, :, :]
            flow_v_blob_slice = flow_v_blob[0, i, :, :]
            flow_h_blob_slice = flow_h_blob[0, i, :, :]
            flow_img[:, :, 0] = (flow_v_blob_slice-(np.mean(flow_v_blob_slice)/2))*2
            flow_img[:, :, 1] = (flow_h_blob_slice-(np.mean(flow_h_blob_slice)/2))*2
            flow_color_img = flow_to_color(flow_img, convert_to_bgr=False) # Change flow as color flow
            for c in range(3):
                sai_img[:, :, c] = cv2.resize(dst_blob_slice[0, c, :, :], dsize=(sai_w, sai_h), interpolation=cv2.INTER_AREA)
            sai_img_list[:, :, :, i] = sai_img
            flow_img_list[:, :, :, i] = flow_color_img

        # Print SAIs and flows
        for i in range(sai_amount):
            cv2.imwrite(pr_path+'/sai'+str(i_tot)+'_'+str(i)+'.png', sai_img_list[:, :, :, i])
            cv2.imwrite(pr_path+'/flow'+str(i_tot)+'_'+str(i)+'.png', flow_img_list[:, :, :, i])

        # Print LF
        sai_img_list2 = trans_order(sai_img_list)
        lf_img = trans_SAIs_to_LF(sai_img_list2)
        cv2.imwrite(pr_path+'/result_lf'+str(i_tot)+'.jpg', lf_img)
        
        # Print Grid and EPI
        sai_mv = np.zeros((sai_h, sai_w, 3, 5, 5))
        sai_epi_ver = np.zeros((sai_h, ang_w*ang_w, 3))
        sai_epi_hor = np.zeros((ang_h*ang_h, sai_w, 3))
        for ax in range(ang_w):
            for ay in range(ang_h):
                sai = sai_img_list[:, :, :, ang_h*ax+ay]
                sai_mv[:, :, :, ay, ax] = sai
                sai_epi_ver[:, ang_w*ax+ay, :] = sai_mv[:, ang_w//2, :, ay, ax]
                sai_epi_hor[ang_h*ax+ay, :, :] = sai_mv[ang_h//2, :, :, ay, ax]
        cv2.imwrite(pr_path+'/sai_epi_ver'+str(i_tot)+'.png', sai_epi_ver)
        cv2.imwrite(pr_path+'/sai_epi_hor'+str(i_tot)+'.png', sai_epi_hor)

        lf_func.make_gird(sai_img_list, (ang_h, ang_w), pr_path+'/sai_grid'+str(i_tot)+'.png')
        
        # Print SAI gif
        lf_func.make_gif(sai_img_list, pr_path+'/sai_gif'+str(i_tot)+'.gif', 0.1)

        SIZE = 256
        OUTPUT_GT = gt_path
        ### Print GT ###
        # Get GT
        sai_GT_list = np.zeros((SIZE, SIZE, 3, 25))
        for i in range(25):
            i_pick = index_picker_5x5(i, args.pick_mode)
            sai_GT = cv2.imread(args.testset_path+'/sai'+str(i_tot)+'_'+str(i_pick)+'.png', 1)
            sai_GT = cv2.resize(sai_GT, dsize=(SIZE, SIZE), interpolation=cv2.INTER_AREA)
            sai_GT_list[:, :, :, i] = sai_GT

        # Print SAIs
        for i in range(25):
            i_pick = index_picker_5x5(i, args.pick_mode)
            cv2.imwrite(OUTPUT_GT+'/sai'+str(i_tot)+'_'+str(i_pick)+'.png', sai_GT_list[:, :, :, i])

        # Print LF
        sai_GT_list2 = trans_order(sai_GT_list)
        lf_GT = trans_SAIs_to_LF(sai_GT_list2)
        cv2.imwrite(OUTPUT_GT+'/result_lf'+str(i_tot)+'.jpg', lf_GT)
        
        # Print Grid and EPI
        sai_GT_uv = np.zeros((SIZE, SIZE, 3, 5, 5))
        sai_GT_grid = np.zeros((SIZE*5, SIZE*5, 3))
        sai_GT_epi_ver = np.zeros((SIZE, 5*5, 3))
        sai_GT_epi_hor = np.zeros((5*5, SIZE, 3))
        for ax in range(5):
            for ay in range(5):
                sai_GT = sai_GT_list[:, :, :, 5*ax+ay]
                sai_GT_uv[:, :, :, ay, ax] = sai_GT
                sai_GT_grid[SIZE*ay:SIZE*ay+SIZE, SIZE*ax:SIZE*ax+SIZE, :] = sai_GT
                sai_GT_epi_ver[:, 5*ax+ay, :] = sai_GT_uv[:, SIZE//2, :, ay, ax]
                sai_GT_epi_hor[5*ax+ay, :, :] = sai_GT_uv[SIZE//2, :, :, ay, ax]
        cv2.imwrite(OUTPUT_GT+'/sai_grid'+str(i_tot)+'.png', sai_GT_grid)
        cv2.imwrite(OUTPUT_GT+'/sai_epi_ver'+str(i_tot)+'.png', sai_GT_epi_ver)
        cv2.imwrite(OUTPUT_GT+'/sai_epi_hor'+str(i_tot)+'.png', sai_GT_epi_hor)

        ### Validation ###
        sai_img_list = sai_img_list.astype('uint8')
        sai_GT_list = sai_GT_list.astype('uint8')


        # PSNR
        def my_psnr(img1, img2):
            mse = np.mean( (img1 - img2) ** 2 )
            if mse == 0:
                return 100
            PIXEL_MAX = 255.0
            return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

        psnr_tot = 0
        for i in range(25):
            #psnr = cv2.PSNR(sai_list[:, :, :, i], sai_GT_list[:, :, :, i])
            psnr = my_psnr(sai_img_list[:, :, :, i], sai_GT_list[:, :, :, i])
            psnr_tot = psnr_tot+psnr
        psnr = psnr_tot / 25

        # SSIM
        ssim_noise_tot = 0
        for i in range(25):
            ssim_noise = skimage.metrics.structural_similarity(sai_img_list[:, :, :, i], sai_GT_list[:, :, :, i], multichannel=True, data_range=sai_img_list[:, :, :, i].max() - sai_img_list[:, :, :, i].min())
            ssim_noise_tot = ssim_noise_tot + ssim_noise
        ssim_noise = ssim_noise_tot / 25

        # Print error map
        lf_func.make_err_map(sai_img_list, sai_GT_list, OUTPUT_GT+'/error_map'+str(i_tot)+'.gif', 0.1, 0.1)

        # Processing Time
        end = time.time()

        # Print Result
        psnr_mean = psnr_mean + psnr
        ssim_mean = ssim_mean + ssim_noise
        time_mean = time_mean + end-start
        print('i : '+str(i_tot)+' | '+'time labs : '+str(end-start)+' | '+'PSNR : '+str(psnr)+' | '+' | '+'SSIM : '+str(ssim_noise))

    # Print Total Result
    psnr_mean = psnr_mean / args.test_size
    ssim_mean = ssim_mean / args.test_size
    time_mean = time_mean / args.test_size
    print('Total result | '+'time labs : '+str(time_mean)+' | '+'PSNR : '+str(psnr_mean)+' | '+' | '+'SSIM : '+str(ssim_mean))

####################################################################################################
##########                                     Main                                       ##########
####################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Light field')

    parser.add_argument('--trainset_path', required=False, default='./datas/face_dataset/face_test_9x9', help='trainset path')
    parser.add_argument('--testset_path', required=False, default='./datas/face_dataset/face_test_9x9', help='testset path')
    parser.add_argument('--train_path', required=False, default='./scripts/denseUNet_train.prototxt', help='train path')
    parser.add_argument('--test_path', required=False, default='./scripts/denseUNet_test.prototxt', help='test path')
    parser.add_argument('--solver_path', required=False, default='./scripts/denseUNet_solver.prototxt', help='solver path')
    parser.add_argument('--model_path', required=False, default= './models/denseUNet', help='model path')
    parser.add_argument('--model_name', required=False, default= './models/denseUNet_iter_10000.caffemodel', help='model name')
    parser.add_argument('--result_path', required=False, default='./output', help='result path')
    parser.add_argument('--train_size', required=False, default=1207, help='train size')
    parser.add_argument('--test_size', required=False, default=98, help='test size')
    parser.add_argument('--n_sai', required=False, default=25, help='num of sai')
    parser.add_argument('--shift_val', required=False, default=0.77625, help='shift value')
    parser.add_argument('--batch_size', required=False, default=4, help='batch size')
    parser.add_argument('--pick_mode', required=False, default='9x9', help='pick mode')
    parser.add_argument('--center_id', required=False, default=12, help='center id')
    parser.add_argument('--epoch', required=False, default=100, help='epoch')
    parser.add_argument('--lr', required=False, default=0.0001, help='learning rate')
    parser.add_argument('--mode', required=False, default='test', help='mode')
    
    args = parser.parse_args()
    
    # Generate Network and Solver    
    if args.mode == 'train': 
        denseUNet_train(args)
        denseUNet_test(args)
        denseUNet_solver(args)
        solver = caffe.get_solver(args.solver_path)
        solver.net.copy_from('./models/denseUNet_iter_9800.caffemodel')
        solver.solve()
    elif args.mode == 'test':
        denseUNet_deploy2(args)
        denseUNet_tester(args)
    elif args.mode == 'deploy':
        denseUNet_deploy2(args)

        

"""
Memo
1. su root
2. ./tools/extra/plot_training_log.py.example 6 /home/junhyeong/docker/Caffe_LF_Syn/plot.png /home/junhyeong/docker/Caffe_LF_Syn/train.log
/opt/caffe/build/tools/caffe train --solver=/docker/Caffe_LF_Syn/scripts/denseUNet_solver.prototxt 2>&1 | tee train.log
"""