import cv2
import numpy as np
import json
import caffe
from caffe.io import caffe_pb2
from caffe import layers as L
from caffe import params as P
import time
from function import *

def image_data_center(batch_size, data_path, center_id, color_mode):
    center, trash = L.ImageData(batch_size=batch_size,
                        source=data_path+'/dataset_list{}.txt'.format(center_id),
                        transform_param=dict(scale=1./256.),
                        shuffle=False,
                        ntop=2,
                        new_height=256,
                        new_width=256,
                        is_color=color_mode)
    silence = L.Silence(trash, ntop=0)
    return center, silence

def image_data_depth(batch_size, data_path, color_mode):
    depth, trash = L.ImageData(batch_size=batch_size,
                        source=data_path+'/depthset_list.txt',
                        transform_param=dict(scale=1.),
                        shuffle=False,
                        ntop=2,
                        new_height=256,
                        new_width=256,
                        is_color=color_mode)
    silence = L.Silence(trash, ntop=0)
    return depth, silence

def image_data(batch_size, data_path, n_sai, color_mode):
    for i_sai in range(n_sai):
        label, trash = L.ImageData(batch_size=batch_size,
                                source=data_path+'/dataset_list{}.txt'.format(i_sai),
                                transform_param=dict(scale=1./256.),
                                shuffle=False,
                                ntop=2,
                                new_height=256,
                                new_width=256,
                                is_color=color_mode)
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

def slice_warp_python(src, flow_h, flow_v, n_sai):
    for i in range(n_sai):
        if i < n_sai-1:
            flow_h_slice, flow_h = L.Slice(flow_h, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
            flow_v_slice, flow_v = L.Slice(flow_v, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
            flow_hv_slice = L.Concat(*[flow_h_slice, flow_v_slice], concat_param={'axis': 1}) # 기존 warping layer랑 반대
            predict_slice = L.Python(*[src, flow_hv_slice], module = 'warping_layer', layer = 'WarpingLayer', ntop = 1)
        else:
            flow_h_slice = flow_h
            flow_v_slice = flow_v
            flow_hv_slice = L.Concat(*[flow_h_slice, flow_v_slice], concat_param={'axis': 1}) # 기존 warping layer랑 반대
            predict_slice = L.Python(*[src, flow_hv_slice], module = 'warping_layer', layer = 'WarpingLayer', ntop = 1)

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

def denseUNet(input, batch_size):
    def flow_layer(bottom=None, nout=1):    
        conv = L.Convolution(bottom, num_output=nout, kernel_size=1, stride=1, dilation=1, pad=0,
                                bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        return conv

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
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True, engine=1)
        conv = L.Convolution(conv, num_output=nout, kernel_size=3, stride=1, dilation=9, pad=9,
                                bias_term=True, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), engine=1)
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=True, engine=1)
        elth = L.Eltwise(conv, bottom, operation=P.Eltwise.SUM)
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
        return conv, pool

    def upsample_concat_layer(bottom1=None, bottom2=None, nout=1):
        deconv = L.Deconvolution(bottom1, convolution_param=dict(num_output=nout, kernel_size=4, stride=2, pad=1))

        #deconv = L.BatchNorm(deconv, use_global_stats=True)
        #deconv = L.Scale(deconv, bias_term=True)
        #bottom2 = L.BatchNorm(bottom2, use_global_stats=True)
        #bottom2 = L.Scale(bottom2, bias_term=True)

        conc = L.Concat(*[deconv, bottom2], concat_param={'axis': 1})
        return conc

    conv1, pool1 = conv_conv_downsample_group_layer(input, 50)
    conv2, pool2 = conv_conv_downsample_group_layer(pool1, 64)
    conv3, pool3 = conv_conv_downsample_group_layer(pool2, 64)
    conv4, pool4 = conv_conv_downsample_group_layer(pool3, 128)
    conv5, pool5 = conv_conv_downsample_group_layer(pool4, 256)
    
    feature = conv_conv_res_dense_layer(pool5, 512)

    deconv1 = upsample_concat_layer(feature, conv5, 256)
    deconv1 = conv_conv_group_layer(deconv1, 256)
    deconv2 = upsample_concat_layer(deconv1, conv4, 128)
    deconv2 = conv_conv_group_layer(deconv2, 128)
    deconv3 = upsample_concat_layer(deconv2, conv3, 64)
    deconv3 = conv_conv_group_layer(deconv3, 64)
    deconv4 = upsample_concat_layer(deconv3, conv2, 64)
    deconv4 = conv_conv_group_layer(deconv4, 64)
    deconv5 = upsample_concat_layer(deconv4, conv1, 50)
    deconv5 = conv_conv_group_layer(deconv5, 50)

    flow = flow_layer(deconv5, 50)

    return flow

def train_proto_gen(args):
    for i_sai in range(args.n_sai):
        i_pick = index_picker_5x5(i_sai, args.pick_mode)
        f = open(args.trainset_path+'/dataset_list{}.txt'.format(i_sai), 'w')
        for i_tot in range(args.train_size):
            f.write(args.trainset_path+'/sai{}_{}.png 0\n'.format(i_tot, i_pick))
        f.close()

    n = caffe.NetSpec()

    n.input, n.silence = image_data_center(batch_size=args.batch_size, data_path=args.trainset_path, center_id=args.center_id, color_mode=False)
    n.label = image_data(batch_size=args.batch_size, data_path=args.trainset_path, n_sai=args.n_sai, color_mode=False)

    n.flow = denseUNet(n.input, args.batch_size)
    n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))

    n.predict = slice_warp2(n.input, n.flow_h, n.flow_v, args.n_sai)

    n.loss1 = L.AbsLoss(n.predict, n.label, loss_weight=1)

    with open(args.train_path, 'w') as f:
        f.write(str(n.to_proto()))    

def test_proto_gen(args):
    if args.mode == 'train':
        for i_sai in range(args.n_sai):
            i_pick = index_picker_5x5(i_sai, args.pick_mode)
            f = open(args.testset_path+'/dataset_list{}.txt'.format(i_sai), 'w')
            for i_tot in range(args.test_size):
                f.write(args.testset_path+'/sai{}_{}.png 0\n'.format(i_tot, i_pick))
            f.close()
 
    n = caffe.NetSpec()

    if args.mode == 'train':
        n.input, n.silence = image_data_center(batch_size=1, data_path=args.testset_path, center_id=args.center_id, color_mode=False)
        n.label = image_data(batch_size=1, data_path=args.testset_path, n_sai=args.n_sai, color_mode=False)
    else:
        n.input = L.Input(input_param=dict(shape=dict(dim=[1, 1, 256, 256])))
        n.input_color = L.Input(input_param=dict(shape=dict(dim=[1, 3, 256, 256])))

    n.flow = denseUNet(n.input, 1)   
    n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[args.n_sai]))

    if args.mode == 'train':
        n.predict = slice_warp2(n.input, n.flow_h, n.flow_v, args.n_sai)

        n.trash = L.Python(n.flow_h, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                        param_str=str(dict(path='./datas/face_dataset', name='flow_h', mult=20)))
        n.silence_flow_h = L.Silence(n.trash, ntop=0)
        n.trash = L.Python(n.flow_v, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                        param_str=str(dict(path='./datas/face_dataset', name='flow_v', mult=20)))
        n.silence_flow_v = L.Silence(n.trash, ntop=0)
        n.trash = L.Python(n.label, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                        param_str=str(dict(path='./datas/face_dataset', name='label', mult=1*256)))
        n.silence_label = L.Silence(n.trash, ntop=0)
        n.trash = L.Python(n.predict, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                        param_str=str(dict(path='./datas/face_dataset', name='predict', mult=1*256)))
        n.silence_predict = L.Silence(n.trash, ntop=0)
    else:
        n.predict = slice_warp_python(n.input_color, n.flow_h, n.flow_v, args.n_sai)

    with open(args.test_path, 'w') as f:
        f.write(str(n.to_proto()))

def solver_proto_gen(args):
    s = caffe_pb2.SolverParameter()

    s.train_net = args.train_path

    if args.test_path is not 'ignore': # None
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