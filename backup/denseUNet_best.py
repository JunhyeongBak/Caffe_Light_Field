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
import skimage

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

def index_picker_5x5(i):
    PICK_MODE = '9x9'
    if PICK_MODE == '9x9':
        id_list = [20, 21, 22, 23, 24,
                    29, 30, 31, 32, 33,
                    38, 39, 40, 41, 42,
                    47, 48, 49, 50, 51,
                    56, 57, 58, 59, 60]
    elif PICK_MODE == '8x8':
        id_list = [9, 10, 11, 12, 13,
                    17, 18, 19, 20, 21,
                    25, 26, 27, 28, 29,
                    33, 34, 35, 36, 37,
                    41, 42, 43, 44, 45]
    return id_list[i]

def image_data_center(batch_size=1, data_path=None, source='train_source'):
    CENTER_ID = 12
    center, trash = L.ImageData(batch_size=batch_size,
                        source=data_path+'/'+source+str(CENTER_ID)+'.txt',
                        transform_param=dict(scale=1./256.),
                        shuffle=False,
                        ntop=2,
                        new_height=256,
                        new_width=256,
                        is_color=False)
    silence = L.Silence(trash, ntop=0)
    return center

def image_data(batch_size=1, data_path=None, source='train_source', sai=25):
    for i_sai in range(sai):
        label, trash = L.ImageData(batch_size=batch_size,
                                source=data_path+'/'+source+str(i_sai)+'.txt',
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
        silence = L.Silence(trash, ntop=0)
    return con

def input_shifting(center=None, batch_size=1, sai=27, shift_val=0.7):
    con = None
    for i_sai in range(sai):
        tx, ty = shift_value_5x5(i_sai, shift_val)
        param_str = json.dumps({'tx': tx, 'ty': ty})
        shift = L.Python(center, module = 'input_shifting_layer', layer = 'InputShiftingLayer', ntop = 1, param_str = param_str)
        if i_sai == 0:
            con = shift
        else:   
            con = L.Concat(*[con, shift], concat_param={'axis': 1})
    return con

def slice_warp(shift, flow_h, flow_v, sai):
    for i in range(sai):
        if i < sai-1:
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

def slice_warp2(shift, flow_h, flow_v, sai):
    for i in range(sai):
        if i < sai-1:
            shift_slice, shift = L.Slice(shift, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
            flow_h_slice, flow_h = L.Slice(flow_h, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
            flow_v_slice, flow_v = L.Slice(flow_v, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
            flow_hv_slice = L.Concat(*[flow_h_slice, flow_v_slice], concat_param={'axis': 1})
            predict_slice = L.FlowWarp(shift_slice, flow_hv_slice)
        else:
            shift_slice = shift
            flow_h_slice = flow_h
            flow_v_slice = flow_v
            flow_hv_slice = L.Concat(*[flow_h_slice, flow_v_slice], concat_param={'axis': 1})
            predict_slice = L.FlowWarp(shift_slice, flow_hv_slice)
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



####################################################################################################
##########                             Fundamental Network                                ##########
####################################################################################################

def denseUNet(input, batch_size):
    def flow_layer(bottom=None, nout=1):
        bottom = L.ReLU(bottom, in_place=True, engine=1)
        flow_init = L.Convolution(bottom, num_output=1, kernel_size=3, stride=1, dilation=1, pad=1,
                                group=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
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
        deconv = L.Deconvolution(bottom1, convolution_param=dict(num_output=nout, kernel_size=3, stride=2, pad=0))
        deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=True)
        dum = L.DummyData(shape=dict(dim=[batch_size, nout, crop_size, crop_size]))
        deconv_crop = L.Crop(deconv, dum, crop_param=dict(axis=1, offset=[0, 0, 0]))
        conc = L.Concat(*[deconv_crop, bottom2], concat_param={'axis': 1})
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

def denseUNet_train(script_path=None, data_path=None, tot=1, sai=25, shift_val=0.7, batch_size=1):
    # Data list txt generating
    for i_sai in range(sai):
        i_pick = index_picker_5x5(i_sai)
        f = open(data_path+'/train_source'+str(i_sai)+'.txt', 'w')
        for i_tot in range(tot):
            data = data_path+'/sai'+str(i_tot)+'_'+str(i_pick)+'.png 0'+'\n'
            f.write(data)
        f.close()

    i_sai = 12
    i_pick = index_picker_5x5(i_sai)
    f = open(data_path+'_depth'+'/train_depth_source'+str(i_sai)+'.txt', 'w')
    for i_tot in range(tot):
        data = data_path+'_depth'+'/sai'+str(i_tot)+'_'+str(i_pick)+'.png 0'+'\n'
        f.write(data)
    f.close()

    # Starting Net    
    n = caffe.NetSpec()

    # Data loading
    n.input = image_data_center(batch_size=batch_size, data_path=data_path, source='train_source')
    n.depth_label = image_data_center(batch_size=batch_size, data_path=data_path+"_depth", source='train_depth_source')
    n.label = image_data(batch_size=batch_size, data_path=data_path, source='train_source', sai=sai)

    # Fundamental Network
    n.flow, n.flow_init = denseUNet(n.input, batch_size)

    # Translation
    n.shift = input_shifting(center=n.input, batch_size=batch_size, sai=sai, shift_val=shift_val)
    n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.predict = slice_warp2(n.shift, n.flow_h, n.flow_v, sai)
    n.depth_predict = L.Convolution(n.flow_init, num_output=1, kernel_size=1, stride=1, dilation=1, pad=0,
                            bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)

    # Loss   
    n.lossA = L.AbsLoss(n.predict, n.label)
    n.lossA = L.Power(n.lossA, power=1.0, scale=1.0, shift=0.0, in_place=True, loss_weight=1)
    #n.lossB = L.AbsLoss(n.depth_predict, n.depth_label)
    #n.lossB = L.Power(n.lossB, power=1.0, scale=0.0, shift=0.0, in_place=True, loss_weight=1)

    n.loss, n.loss2 = l_loss(n.predict, n.label, batch_size)
    n.loss = L.Power(n.loss, power=1.0, scale=1.0, shift=0.0, in_place=True, loss_weight=1)
    n.loss2 = L.Power(n.loss2, power=1.0, scale=1.0, shift=0.0, in_place=True, loss_weight=1)

    # Generating Prototxt
    with open(script_path, 'w') as f:
        f.write(str(n.to_proto()))    

def denseUNet_test(script_path=None, data_path=None, tot=1, sai=25, shift_val=0.7, batch_size=1):
    # Data list txt generating
    for i_sai in range(sai):
        i_pick = index_picker_5x5(i_sai)
        f = open(data_path+'/test_source'+str(i_sai)+'.txt', 'w')
        for i_tot in range(tot):
            data = data_path+'/sai'+str(i_tot)+'_'+str(i_pick)+'.png 0'+'\n'
            f.write(data)
        f.close()

    # Starting Net     
    n = caffe.NetSpec()

    # Data Loading
    n.input = image_data_center(batch_size=batch_size, data_path=data_path, source='test_source')
    n.label = image_data(batch_size=batch_size, data_path=data_path, source='test_source', sai=sai)

    # Fundamental Network
    n.flow, n.flow_init = denseUNet(n.input, batch_size)   

    # Translation
    n.shift = input_shifting(center=n.input, batch_size=batch_size, sai=sai, shift_val=shift_val)
    n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    param_str = json.dumps({'flow_size': 25, 'color_size': 1})
    n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1, param_str = param_str)
    n.depth_predict = L.Convolution(n.flow_init, num_output=1, kernel_size=1, stride=1, dilation=1, pad=0,
                            bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)

    n.label_order = change_order(n.label, batch_size)
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
    n.trash6 = L.Python(n.depth_predict, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset', name='depth_predict', mult=1*256)))
    n.silence6 = L.Silence(n.trash6, ntop=0)
    n.trash7 = L.Python(n.label_order, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset', name='label_order', mult=1*256)))
    n.silence7 = L.Silence(n.trash7, ntop=0)

    # Generating Prototxt
    with open(script_path, 'w') as f:
        f.write(str(n.to_proto()))

def denseUNet_deploy(script_path=None, batch_size=1, sai=25, shift_val=0.7):
    n = caffe.NetSpec()

    # Data loading
    n.input = L.Input(input_param=dict(shape=dict(dim=[1, 3, 256, 256])))
    n.input_luma = L.Input(input_param=dict(shape=dict(dim=[1, 1, 256, 256])))
    n.input_luma_down = L.Power(n.input_luma, power=1.0, scale=1./256., shift=0, in_place=False)
    n.shift = input_shifting(center=n.input, batch_size=batch_size, sai=sai, shift_val=shift_val)

    # Network
    n.flow, n.flow_init = denseUNet(n.input_luma_down, batch_size)
    n.silence = L.Silence(n.flow_init, ntop=0)  

    # Translation
    n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    param_str = json.dumps({'flow_size': 25, 'color_size': 3})
    n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1, param_str = param_str)

    with open(script_path, 'w') as f:
        f.write(str(n.to_proto()))

####################################################################################################
##########                                Solver Generator                                ##########
####################################################################################################
    
def denseUNet_solver(script_train_path=None, script_test_path=None, solver_path=None, snapshot_path=None):
    s = caffe_pb2.SolverParameter()

    s.train_net = script_train_path
    if script_test_path is not None:
        s.test_net.append(script_test_path)
        s.test_interval = 20
        s.test_iter.append(1)
    else:
        s.test_initialization = False
    
    s.iter_size = 1
    s.max_iter = 500000
    s.type = 'Adam'
    s.base_lr = 0.001 # 0.0005,  0.0000001 # 0.000005(basic), 0.0000001
    s.lr_policy = 'step'
    s.gamma = 0.75
    s.power = 0.75
    s.stepsize = 1000
    s.momentum = 0.999
    s.momentum2 = 0.999
    s.weight_decay = 0.0005
    s.clip_gradients = 1
    s.display = 25

    s.snapshot = 1000

    if snapshot_path is not None:
        s.snapshot_prefix = snapshot_path

    s.solver_mode = caffe_pb2.SolverParameter.GPU

    with open(solver_path, 'w') as f:
        f.write(str(s))

####################################################################################################
##########                                      Runner                                    ##########
####################################################################################################

def denseUNet_runner(script_path=None, model_path=None, src_color=None, n=None):
    # Set a model
    if n == None:
        n = caffe.Net(script_path, model_path, caffe.TEST)

    # Input images    
    src_color = cv2.resize(src_color, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    src_luma = cv2.cvtColor(src_color, cv2.COLOR_BGR2GRAY)
    src_blob_color = np.zeros((1, 3, 256, 256))
    src_blob_luma = np.zeros((1, 1, 256, 256))
    for i in range(3):
        src_blob_color[0, i, :, :] = src_color[:, :, i]
    src_blob_luma[0, 0, :, :] = src_luma[:, :]
    n.blobs['input'].data[...] = src_blob_color
    n.blobs['input_luma'].data[...] = src_blob_luma

    # Net forward        
    n.forward()

    # Get and print sai
    sai = np.zeros((256, 256, 3))
    sai_list =  np.zeros((256, 256, 3, 5*5))
    flow = np.zeros((256, 256, 2))
    flow_color_list = np.zeros((256, 256, 3, 5*5))
    dst_blob = n.blobs['predict'].data[...]
    flow_v_blob = n.blobs['flow_v'].data[...]
    flow_h_blob = n.blobs['flow_h'].data[...]
    for i in range(25):
        dst_blob_slice = dst_blob[:, 3*i:3*i+3, :, :]
        flow_v_blob_slice = flow_v_blob[0, i, :, :]
        flow_h_blob_slice = flow_h_blob[0, i, :, :]
        flow[:, :, 0] = (flow_v_blob_slice-(np.mean(flow_v_blob_slice)/2))*2
        flow[:, :, 1] = (flow_h_blob_slice-(np.mean(flow_h_blob_slice)/2))*2

        flow_color = flow_to_color(flow, convert_to_bgr=False)
        for c in range(3):
            sai[:, :, c] = cv2.resize(dst_blob_slice[0, c, :, :], dsize=(256, 256), interpolation=cv2.INTER_AREA)
        sai_list[:, :, :, i] = sai
        flow_color_list[:, :, :, i] = flow_color
        #cv2.imwrite(PATH+'/output/sai'+str(i_tot)+'_'+str(i)+'.png', sai)

    return sai_list, flow_color_list

####################################################################################################
##########                                    Tester                                      ##########
####################################################################################################

def denseUNet_tester(script_path=None,
                        model_path=None,
                        dataset_path=None,
                        output_predict_path=None,
                        output_GT_path=None,
                        test_range=1):
    SIZE = 256

    # Set a model
    n = caffe.Net(script_path, model_path, caffe.TEST)

    # Iterate
    psnr_mean = 0
    ssim_mean = 0
    time_mean = 0
    for i_tot in range(test_range):
        start = time.time()

        ##### Predict Process #####
        # Run Model
        i_pick = index_picker_5x5(12)
        src_img = cv2.imread(dataset_path+'/sai'+str(i_tot)+'_'+str(i_pick)+'.png')
        src_img = cv2.resize(src_img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        sai_list, flow_color_list = denseUNet_runner(script_path=None, model_path=None, src_color=src_img, n=n)

        # Print SAIs and flows
        for i in range(25):
            cv2.imwrite(output_predict_path+'/sai'+str(i_tot)+'_'+str(i)+'.png', sai_list[:, :, :, i])
            cv2.imwrite(output_predict_path+'/flow'+str(i_tot)+'_'+str(i)+'.png', flow_color_list[:, :, :, i])

        # Print LF
        sai_list2 = trans_order(sai_list)
        lf = trans_SAIs_to_LF(sai_list2)
        cv2.imwrite(output_predict_path+'/result_lf'+str(i_tot)+'.jpg', lf)
        
        # Print Grid and EPI
        sai_uv = np.zeros((SIZE, SIZE, 3, 5, 5))
        sai_grid = np.zeros((SIZE*5, SIZE*5, 3))
        sai_epi_ver = np.zeros((SIZE, 5*5, 3))
        sai_epi_hor = np.zeros((5*5, SIZE, 3))
        for ax in range(5):
            for ay in range(5):
                sai = sai_list[:, :, :, 5*ax+ay]
                sai_uv[:, :, :, ay, ax] = sai
                sai_grid[SIZE*ay:SIZE*ay+SIZE, SIZE*ax:SIZE*ax+SIZE, :] = sai
                sai_epi_ver[:, 5*ax+ay, :] = sai_uv[:, SIZE//2, :, ay, ax]
                sai_epi_hor[5*ax+ay, :, :] = sai_uv[SIZE//2, :, :, ay, ax]
        cv2.imwrite(output_predict_path+'/sai_grid'+str(i_tot)+'.png', sai_grid)
        cv2.imwrite(output_predict_path+'/sai_epi_ver'+str(i_tot)+'.png', sai_epi_ver)
        cv2.imwrite(output_predict_path+'/sai_epi_hor'+str(i_tot)+'.png', sai_epi_hor)

        ##### GT Process #####
        # Get GT
        sai_GT_list = np.zeros((SIZE, SIZE, 3, 25))
        for i in range(25):
            i_pick = index_picker_5x5(i)
            sai_GT = cv2.imread(dataset_path+'/sai'+str(i_tot)+'_'+str(i_pick)+'.png')
            sai_GT = cv2.resize(sai_GT, dsize=(SIZE, SIZE), interpolation=cv2.INTER_AREA)
            sai_GT_list[:, :, :, i] = sai_GT
        
        # Print LF
        sai_GT_list2 = trans_order(sai_GT_list)
        lf_GT = trans_SAIs_to_LF(sai_GT_list2)
        cv2.imwrite(output_GT_path+'/result_lf'+str(i_tot)+'.jpg', lf_GT)
        
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
        cv2.imwrite(output_GT_path+'/sai_grid'+str(i_tot)+'.png', sai_GT_grid)
        cv2.imwrite(output_GT_path+'/sai_epi_ver'+str(i_tot)+'.png', sai_GT_epi_ver)
        cv2.imwrite(output_GT_path+'/sai_epi_hor'+str(i_tot)+'.png', sai_GT_epi_hor)

        ##### Validation Process #####
        sai_list = sai_list.astype('uint8')
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
            psnr = my_psnr(sai_list[:, :, :, i], sai_GT_list[:, :, :, i])
            psnr_tot = psnr_tot+psnr
        psnr = psnr_tot / 25

        # SSIM
        ssim_noise_tot = 0
        for i in range(25):
            ssim_noise = skimage.metrics.structural_similarity(sai_list[:, :, :, i], sai_GT_list[:, :, :, i], multichannel=True, data_range=sai_list[:, :, :, i].max() - sai_list[:, :, :, i].min())
            ssim_noise_tot = ssim_noise_tot + ssim_noise
        ssim_noise = ssim_noise_tot / 25

        # Processing Time
        end = time.time()

        # Print Result
        psnr_mean = psnr_mean + psnr
        ssim_mean = ssim_mean + ssim_noise
        time_mean = time_mean + end-start
        print('i : '+str(i_tot)+' | '+'time labs : '+str(end-start)+' | '+'PSNR : '+str(psnr)+' | '+' | '+'SSIM : '+str(ssim_noise))

    # Print Total Result
    psnr_mean = psnr_mean / test_range
    ssim_mean = ssim_mean / test_range
    time_mean = time_mean / test_range
    print('Total result | '+'time labs : '+str(time_mean)+' | '+'PSNR : '+str(psnr_mean)+' | '+' | '+'SSIM : '+str(ssim_mean))

####################################################################################################
##########                                     Main                                       ##########
####################################################################################################

if __name__ == "__main__":
    # Constant
    TRAINSET_PATH = './datas/face_dataset/face_train_9x9'
    TESTSET_PATH = './datas/face_dataset/face_test_9x9'
    MODEL_PATH = './backup/denseUNet_best.caffemodel'
    TRAIN_PATH = './scripts/denseUNet_train.prototxt'
    TEST_PATH = './scripts/denseUNet_test.prototxt'
    DEPLOY_PATH = './scripts/denseUNet_deploy.prototxt'
    SOLVER_PATH = './scripts/denseUNet_solver.prototxt'
    OUTPUT_PREDICT = './output/predict'
    OUTPUT_GT = './output/GT'

    TRAIN_TOT = 1207
    SAI = 25
    SHIFT_VAL = 0.77625 #-1.4
    MOD = 'test'
    
    # Generate Network and Solver
    denseUNet_train(script_path=TRAIN_PATH, data_path=TRAINSET_PATH, tot=TRAIN_TOT, sai=SAI, batch_size=4, shift_val=SHIFT_VAL)
    denseUNet_test(script_path=TEST_PATH, data_path=TESTSET_PATH, tot=TRAIN_TOT, sai=SAI, batch_size=1, shift_val=SHIFT_VAL)
    denseUNet_deploy(script_path=DEPLOY_PATH, batch_size=1, sai=SAI, shift_val=SHIFT_VAL)
    denseUNet_solver(script_train_path=TRAIN_PATH, script_test_path=TEST_PATH, solver_path=SOLVER_PATH, snapshot_path='./models')

    if MOD == 'train': 
        solver = caffe.get_solver(SOLVER_PATH)
        solver.net.copy_from(MODEL_PATH)
        solver.solve()
    elif MOD == 'test':
        denseUNet_tester(script_path=DEPLOY_PATH,
                            model_path=MODEL_PATH,
                            dataset_path=TESTSET_PATH,
                            output_predict_path=OUTPUT_PREDICT,
                            output_GT_path=OUTPUT_GT,
                            test_range=100)
    elif MOD == 'run':
        test_img = cv2.imread('./test.png', 1)
        test_img = cv2.resize(test_img, (256, 256), interpolation=cv2.INTER_AREA)
        test_list, flow_color_list = denseUNet_runner(script_path=DEPLOY_PATH, model_path=MODEL_PATH, src_color=test_img, n=None)
        for i in range(25):
            cv2.imwrite('./sai2_'+str(i)+'.png', test_list[:, :, :, i])
            cv2.imwrite('./flow2_'+str(i)+'.png', flow_color_list[:, :, :, i])
    else:
        print('Network Generated!!!')

"""
Memo
1. su root
2. ./tools/extra/plot_training_log.py.example 6 /home/junhyeong/docker/Caffe_LF_Syn/plot.png /home/junhyeong/docker/Caffe_LF_Syn/train.log
/opt/caffe/build/tools/caffe train --solver=/docker/Caffe_LF_Syn/scripts/denseUNet_solver.prototxt 2>&1 | tee train.log
"""
