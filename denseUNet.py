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
                        transform_param=dict(scale=1.),
                        shuffle=False,
                        ntop=2,
                        new_height=256,
                        new_width=256,
                        is_color=True)
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
            flow_h_slice, flow_h = L.Slice(flow_h, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
            flow_v_slice, flow_v = L.Slice(flow_v, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
            flow_hv_slice = L.Concat(*[flow_h_slice, flow_v_slice], concat_param={'axis': 1})
            predict_slice = L.FlowWarp(src, flow_hv_slice)
        else:
            flow_h_slice = flow_h
            flow_v_slice = flow_v
            flow_hv_slice = L.Concat(*[flow_h_slice, flow_v_slice], concat_param={'axis': 1})
            predict_slice = L.FlowWarp(src, flow_hv_slice)
        if i == 0:
            con = predict_slice
        else:
            con = L.Concat(*[con, predict_slice], concat_param={'axis': 1})
    return con

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

def l_loss(imgs, lables, batch_size=1):
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
            #imgs2_slice = imgs2
            lables_slice = lables
            #lables2_slice = lables2

        imgs_redu = L.Reduction(imgs_slice, axis=1, operation=P.Reduction.MEAN)
        #imgs2_redu = L.Reduction(imgs2_slice, axis=1, operation=P.Reduction.MEAN)
        lables_redu = L.Reduction(lables_slice, axis=1, operation=P.Reduction.MEAN)
        #lables2_redu = L.Reduction(lables2_slice, axis=1, operation=P.Reduction.MEAN)
        
        if i == 0:
            loss = L.AbsLoss(imgs_redu, lables_redu)
            loss2 = L.AbsLoss(imgs2_redu, lables2_redu)
        else:
            temp_loss = L.AbsLoss(imgs_redu, lables_redu)
            temp_loss2 = L.AbsLoss(imgs2_redu, lables2_redu)
            loss = L.Eltwise(loss, temp_loss, operation=P.Eltwise.SUM)
            loss2 = L.Eltwise(loss2, temp_loss2, operation=P.Eltwise.SUM)

    return loss

def org_loss(imgs):
    row1, row2, row3, row4, row5 = L.Slice(imgs, ntop=5, slice_param=dict(slice_dim=1, slice_point=[5, 10, 15, 20]))
   
    row_mean1 = L.Reduction(row1, axis=1, operation=P.Reduction.MEAN)
    row_mean2 = L.Reduction(row2, axis=1, operation=P.Reduction.MEAN)
    row_mean3 = L.Reduction(row3, axis=1, operation=P.Reduction.MEAN)
    row_mean4 = L.Reduction(row4, axis=1, operation=P.Reduction.MEAN)
    row_mean5 = L.Reduction(row5, axis=1, operation=P.Reduction.MEAN)

    return row_mean1, row_mean2, row_mean3, row_mean4, row_mean5

####################################################################################################
##########                             Fundamental Network                                ##########
####################################################################################################

def denseUNet(input, batch_size):
    def flow_layer(bottom=None, nout=1):
        bottom = L.ReLU(bottom, in_place=True, engine=1)
        flow_init = L.Convolution(bottom, num_output=1, kernel_size=1, stride=1, dilation=1, pad=0,
                                group=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        flow = L.Convolution(flow_init, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                group=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        return flow, flow_init

    def conv_conv_group_layer(bottom=None, nout=1):
        conv = L.Convolution(bottom, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                bias_term=True, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True, engine=1)
        conv = L.Convolution(conv, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                group=nout, bias_term=True, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        conv = L.Convolution(conv, num_output=nout, kernel_size=1, stride=1, dilation=1, pad=0,
                                bias_term=True, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True, engine=1)
        return conv

    def conv_conv_res_dense_layer(bottom=None, nout=1):
        bottom = L.Convolution(bottom, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                bias_term=True, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        bottom = L.ReLU(bottom, relu_param=dict(negative_slope=0.2), in_place=True, engine=1)
        conv = L.Convolution(bottom, num_output=nout, kernel_size=3, stride=1, dilation=3, pad=3,
                                bias_term=True, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True, engine=1)
        conv = L.Convolution(conv, num_output=nout, kernel_size=3, stride=1, dilation=6, pad=6,
                                bias_term=True, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        elth = L.Eltwise(conv, bottom, operation=P.Eltwise.SUM)
        elth = L.ReLU(elth, relu_param=dict(negative_slope=0.2), in_place=True, engine=1)
        return elth   

    def conv_conv_downsample_group_layer(bottom=None, nout=1):
        conv = L.Convolution(bottom, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                bias_term=True, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True, engine=1)
        conv = L.Convolution(conv, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                group=nout, bias_term=True, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        conv = L.Convolution(conv, num_output=nout, kernel_size=1, stride=1, dilation=1, pad=0,
                                bias_term=True, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
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

    conv1, poo1 = conv_conv_downsample_group_layer(input, 50)
    conv2, poo2 = conv_conv_downsample_group_layer(poo1, 64)
    conv3, poo3 = conv_conv_downsample_group_layer(poo2, 64)
    conv4, poo4 = conv_conv_downsample_group_layer(poo3, 128)
    conv5, poo5 = conv_conv_downsample_group_layer(poo4, 256)
    
    feature = conv_conv_res_dense_layer(poo5, 512)

    deconv1 = upsample_concat_layer(feature, conv5, 256, 16)
    deconv1 = conv_conv_group_layer(deconv1, 256)
    deconv2 = upsample_concat_layer(deconv1, conv4, 128, 32)
    deconv2 = conv_conv_group_layer(deconv2, 128)
    deconv3 = upsample_concat_layer(deconv2, conv3, 64, 64)
    deconv3 = conv_conv_group_layer(deconv3, 64)
    deconv4 = upsample_concat_layer(deconv3, conv2, 64, 128)
    deconv4 = conv_conv_group_layer(deconv4, 64)
    deconv5 = upsample_concat_layer(deconv4, conv1, 50, 256)
    deconv5 = conv_conv_group_layer(deconv5, 50)

    # flow, flow_init = flow_layer(deconv5, 25*2)

    return deconv5

####################################################################################################
##########                               Network Generator                                ##########
####################################################################################################

def denseUNet_train(args):
    # Make data list
    for i_sai in range(args.n_sai):
        i_pick = index_picker_5x5(i_sai, args.pick_mode)
        f = open(args.trainset_path+'/dataset_list'+str(i_sai)+'.txt', 'w')
        for i_tot in range(args.train_size):
            data = args.trainset_path+'/sai'+str(i_tot)+'_'+str(i_pick)+'.png 0'+'\n'
            f.write(data)
        f.close()

    f = open(args.trainset_path+'/depthset_list.txt', 'w')
    for i_tot in range(args.train_size):
        data = args.trainset_path+'/sai'+str(i_tot)+'_dep.png 0'+'\n' # !!! depth center id = 40 !!!
        f.write(data)
    f.close()

    # Init network   
    n = caffe.NetSpec()

    # Input
    n.input = image_data_center(batch_size=args.batch_size, data_path=args.trainset_path, center_id=args.center_id)
    n.label = image_data(batch_size=args.batch_size, data_path=args.trainset_path, n_sai=args.n_sai)
    #n.label_depth_org = image_data_depth(batch_size=args.batch_size, data_path=args.trainset_path)

    #n.label_depth_1ch, n.label_depth_2ch, n.depth_trash = L.Slice(n.label_depth_org, ntop=3, slice_param=dict(slice_dim=1, slice_point=[1, 2]))
    #n.label_depth_1ch = L.Power(n.label_depth_1ch, power=1.0, scale=256.0, shift=0.0)
    #n.label_depth = L.Eltwise(n.label_depth_1ch, n.label_depth_2ch, operation=P.Eltwise.SUM)
    #n.label_depth = L.Power(n.label_depth, power=1.0, scale=1/(256.0*256.0), shift=0.0)
    #n.depth_trash = L.Silence(n.depth_trash, ntop=0)

    # Fundamental
    n.flow = denseUNet(n.input, args.batch_size)

    # Translation
    n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.predict = slice_warp2(n.input, n.flow_h, n.flow_v, args.n_sai)

    n.loss1 = L.AbsLoss(n.predict, n.label, loss_weight=1)
    #n.loss1 = L.Power(n.loss1, power=1.0, scale=1.0, shift=0.0, loss_weight=1)

    #n.dep = L.Convolution(n.flow, num_output=1, kernel_size=3, stride=1, dilation=1, pad=1,
    #                        group=1, bias_term=True, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
    #n.dep = L.Sigmoid(n.dep, in_place=True)
    #n.dep = L.Convolution(n.dep, num_output=1, kernel_size=1, stride=1, dilation=1, pad=0,
    #                        group=1, bias_term=True, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
    #n.dep = L.Sigmoid(n.dep, in_place=True)
    #n.loss2 = L.AbsLoss(n.dep, n.label_depth, loss_weight=1)
    
    #n.loss2 = L.AbsLoss(n.label_depth, n.flow_init)
    #n.loss2 = L.EuclideanLoss(n.label_depth, n.flow_init)
    #n.loss2 = L.Power(n.loss2, power=1.0, scale=0.3, shift=0.0, loss_weight=1)

    #n.ex_prdict = L.Convolution(n.flow_init, num_output=1, kernel_size=3, stride=1, weight_filler=dict(type='EdgeX'), engine=1)
    #n.ex_label = L.Convolution(n.label_depth, num_output=1, kernel_size=3, stride=1, weight_filler=dict(type='EdgeY'), engine=1)
    #n.loss3 = L.AbsLoss(n.ex_prdict, n.ex_label)
    #n.loss3 = L.Power(n.loss3, power=1.0, scale=1.5, shift=0.0, loss_weight=1)

    # Generate Prototxt
    with open(args.train_path, 'w') as f:
        f.write(str(n.to_proto()))    

def denseUNet_test(args):
    # Generate dataset list
    # Same as train

    # Init network       
    n = caffe.NetSpec()

    # Input
    n.input = image_data_center(batch_size=1, data_path=args.trainset_path, center_id=args.center_id)
    n.label = image_data(batch_size=1, data_path=args.trainset_path, n_sai=args.n_sai)
    #n.label_depth = image_data_depth(batch_size=1, data_path=args.trainset_path)

    # Fundamental
    n.flow = denseUNet(n.input, 1)   

    # Translation
    n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.predict = slice_warp2(n.input, n.flow_h, n.flow_v, args.n_sai)
    
    # Visualization
    n.trash1 = L.Python(n.flow_h, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset', name='flow_h', mult=20)))
    n.silence1 = L.Silence(n.trash1, ntop=0)
    n.trash2 = L.Python(n.flow_v, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset', name='flow_v', mult=20)))
    n.silence2 = L.Silence(n.trash2, ntop=0)
    n.trash3 = L.Python(n.label, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset', name='label', mult=1*256)))
    n.silence3 = L.Silence(n.trash3, ntop=0)
    n.trash4 = L.Python(n.predict, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset', name='predict', mult=1*256)))
    n.silence4 = L.Silence(n.trash4, ntop=0)
    #n.trash5 = L.Python(n.flow_init, module='print_scaled', layer='PrintScaledLayer', ntop=1,
    #                param_str=str(dict(path='./datas/face_dataset/depth_predict.png')))
    #n.silence5 = L.Silence(n.trash5, ntop=0)
    #n.flow_init2 = L.Power(n.flow_init, power=1.0, scale=256*256, shift=0.0)
    #n.trash6 = L.Python(n.flow_init2, module='depth_layer', layer='DepthLayer', ntop=1,
    #                param_str=str(dict(path='./datas/face_dataset/depth_predict_3ch.png')))
    #n.silence6 = L.Silence(n.trash6, ntop=0)

    # Generate Prototxt
    with open(args.test_path, 'w') as f:
        f.write(str(n.to_proto()))

def denseUNet_deploy2(args):
    # Init network  
    n = caffe.NetSpec()

    # Input
    if args.mode == 'deploy':
        n.input, n.trash = L.ImageData(batch_size=1,
                            source='./input_list.txt',
                            shuffle=False,
                            ntop=2,
                            new_height=256,
                            new_width=256,
                            is_color=True)
        n.input_luma, n.trash = L.ImageData(batch_size=1,
                            source='./input_list.txt',
                            shuffle=False,
                            ntop=2,
                            new_height=256,
                            new_width=256,
                            is_color=False)
        n.input_luma_down = L.Power(n.input_luma, power=1.0, scale=1./256., shift=0)
    else:
        n.input = L.Input(input_param=dict(shape=dict(dim=[1, 3, 256, 256])))
        n.input_luma = L.Input(input_param=dict(shape=dict(dim=[1, 1, 256, 256])))
        n.input_luma_down = L.Power(n.input_luma, power=1.0, scale=1./256., shift=0)

    # Fundamental
    n.flow = denseUNet(n.input_luma_down, batch_size=1)
    #n.silence = L.Silence(n.flow_init, ntop=0)  

    # Translation
    n.shift = input_shifting(src=n.input, n_sai=args.n_sai, shift_val=args.shift_val*1)
    n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[args.n_sai]))
    n.flow_h = L.Power(n.flow_h, power=1.0, scale=1, shift=0.0) 
    n.flow_v = L.Power(n.flow_v, power=1.0, scale=1, shift=0.0) 
    n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    param_str = json.dumps({'flow_size': args.n_sai, 'color_size': 3})
    n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1, param_str = param_str)

    # Visualization
    n.result = flat(n.predict, batch_size=1, ch=3)

    n.trash1 = L.Python(n.result, module='print_layer', layer='PrintLayer', ntop=1,
                    param_str=str(dict(path='./', name='mv_result', mult=1)))
    n.silence1 = L.Silence(n.trash1, ntop=0)
    #n.trash2 = L.Python(n.flow_init, module='print_scaled', layer='PrintScaledLayer', ntop=1,
    #                param_str=str(dict(path='./disparity.png')))
    #n.silence2 = L.Silence(n.trash2, ntop=0)

    # Generate Prototxt
    with open('./deploy.prototxt', 'w') as f:
        f.write(str(n.to_proto()))

def sandbox():
    n = caffe.NetSpec()

    n.image, n.trash = L.ImageData(batch_size=1,
                                source='./sandbox.txt',
                                shuffle=False,
                                ntop=2,
                                new_height=256,
                                new_width=256,
                                is_color=True)

    with open('./sandbox.prototxt', 'w') as f:
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
    s.stepsize = 50000
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

def denseUNet_infer():
    n = caffe.Net('./deploy.prototxt', './models/denseUNet_iter_21000.caffemodel', caffe.TEST)

    # Load image
    src_img = cv2.imread('infer_input.png', 1)
    if src_img.all() == None:
        src_img = cv2.imread('infer_input.jpg', 1)
    src_img = cv2.resize(src_img, (256, 256), interpolation=cv2.INTER_AREA)
    luma_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    # Input image
    src_blob = lf_func.img_to_blob(src_img)
    luma_blob = lf_func.img_to_blob(luma_img)
    n.blobs['input'].data[...] = src_blob
    n.blobs['input_luma'].data[...] = luma_blob

    # Run model
    n.forward()

    sai_img_list =  np.zeros((256, 256, 3, 25))
    dst_blob = n.blobs['predict'].data[...]
    dst_blob = np.clip(dst_blob, 0, 255)
    for i in range(25):
        dst_blob_slice = dst_blob[:, 3*i:3*i+3, :, :]
        sai_img_list[:, :, :, i] = lf_func.blob_to_img(dst_blob_slice)

    lf_func.make_gif2(sai_img_list, './infer_result2.gif', 0.1)

def denseUNet_tester(args):
    import skimage

    # Params
    sai_w = 256
    sai_h = 256
    ang_w = 5
    ang_h = 5
    sai_amount = 25
    id_center = 12

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
        id_pick = index_picker_5x5(id_center, pick_mode)
        src_img = cv2.imread(args.testset_path+'/sai'+str(i_tot)+'_'+str(id_pick)+'.png', 1)
        src_img = cv2.resize(src_img, (sai_w, sai_h), interpolation=cv2.INTER_AREA)
        luma_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

        # Input image
        src_blob = lf_func.img_to_blob(src_img)
        luma_blob = lf_func.img_to_blob(luma_img)
        n.blobs['input'].data[...] = src_blob
        n.blobs['input_luma'].data[...] = luma_blob

        # Run model
        n.forward()

        ### Print predict ###
        # Get SAI, flow list
        sai_img_list =  np.zeros((sai_h, sai_w, 3, sai_amount))
        flow_img_list = np.zeros((sai_h, sai_w, 3, sai_amount))
        flow_img = np.zeros((sai_h, sai_w, 2))

        dst_blob = n.blobs['predict'].data[...]
        flow_v_blob = n.blobs['flow_v'].data[...]
        flow_h_blob = n.blobs['flow_h'].data[...]
        for i in range(sai_amount):
            dst_blob_slice = dst_blob[:, 3*i:3*i+3, :, :]
            sai_img = lf_func.blob_to_img(dst_blob_slice)
            sai_img = cv2.resize(sai_img, dsize=(sai_w, sai_h), interpolation=cv2.INTER_AREA)
            sai_img_list[:, :, :, i] = sai_img

            flow_v_blob_slice = flow_v_blob[0, i, :, :]
            flow_h_blob_slice = flow_h_blob[0, i, :, :]
            flow_img[:, :, 0] = (flow_v_blob_slice-(np.mean(flow_v_blob_slice)/2))*2
            flow_img[:, :, 1] = (flow_h_blob_slice-(np.mean(flow_h_blob_slice)/2))*2
            flow_color_img = lf_func.flow_to_color(flow_img, convert_to_bgr=False)
            flow_img_list[:, :, :, i] = flow_color_img
        for i in range(sai_amount):
            cv2.imwrite(pr_path+'/sai'+str(i_tot)+'_'+str(i)+'.png', sai_img_list[:, :, :, i])
            cv2.imwrite(pr_path+'/flow'+str(i_tot)+'_'+str(i)+'.png', flow_img_list[:, :, :, i])

        # Print LF
        sai_img_list2 = lf_func.trans_order(sai_img_list)
        lf_img = lf_func.trans_SAIs_to_LF(sai_img_list2)
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
        lf_func.make_gif2(sai_img_list, pr_path+'/sai_gif'+str(i_tot)+'.gif', 0.1)

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
        sai_GT_list2 = lf_func.trans_order(sai_GT_list)
        lf_GT = lf_func.trans_SAIs_to_LF(sai_GT_list2)
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
        lf_func.make_err_map(sai_img_list, sai_GT_list, OUTPUT_GT+'/error_map'+str(i_tot)+'.gif', 0.5, 0.1)

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

    parser.add_argument('--trainset_path', required=False, default='./datas/face_dataset/face_train_9x9', help='trainset path')
    parser.add_argument('--testset_path', required=False, default='./datas/face_dataset/face_train_9x9', help='testset path')
    parser.add_argument('--train_path', required=False, default='./scripts/denseUNet_train.prototxt', help='train path')
    parser.add_argument('--test_path', required=False, default='./scripts/denseUNet_test.prototxt', help='test path')
    parser.add_argument('--solver_path', required=False, default='./scripts/denseUNet_solver.prototxt', help='solver path')
    parser.add_argument('--model_path', required=False, default= './models/denseUNet', help='model path')
    parser.add_argument('--model_name', required=False, default= './models/denseUNet_iter_21000.caffemodel', help='model name')
    parser.add_argument('--result_path', required=False, default='./output', help='result path')
    parser.add_argument('--train_size', required=False, default=190, help='train size')
    parser.add_argument('--test_size', required=False, default=190, help='test size')
    parser.add_argument('--n_sai', required=False, default=25, help='num of sai')
    parser.add_argument('--shift_val', required=False, default=0, help='shift value')
    parser.add_argument('--batch_size', required=False, default=4, help='batch size')
    parser.add_argument('--pick_mode', required=False, default='9x9', help='pick mode')
    parser.add_argument('--center_id', required=False, default=12, help='center id')
    parser.add_argument('--epoch', required=False, default=100000, help='epoch')
    parser.add_argument('--lr', required=False, default=0.0005, help='learning rate')
    parser.add_argument('--mode', required=False, default='train', help='mode')
    
    args = parser.parse_args()
    
    # Generate Network and Solver    
    if args.mode == 'train': 
        denseUNet_train(args)
        denseUNet_test(args)
        denseUNet_solver(args)
        solver = caffe.get_solver(args.solver_path)
        solver.net.copy_from('./models/denseUNet_iter_21000.caffemodel')
        solver.solve()
    elif args.mode == 'test':
        denseUNet_deploy2(args)
        denseUNet_tester(args)
    elif args.mode == 'deploy':
        denseUNet_deploy2(args)
    elif args.mode == 'infer':
        denseUNet_deploy2(args)
        denseUNet_infer()
    elif args.mode == 'sandbox':
        sandbox()
        denseUNet_solver(args)
        solver = caffe.get_solver(args.solver_path)
        solver.solve()