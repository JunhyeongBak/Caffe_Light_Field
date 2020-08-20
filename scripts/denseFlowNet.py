import os
import sys
import cv2
import numpy as np
import json
import caffe
from caffe.io import caffe_pb2
from caffe import layers as L
from caffe import params as P

caffe.set_mode_gpu()
np.set_printoptions(threshold=sys.maxsize)

CENTER_SAI = 12

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

if __name__ == "__main__":
    def shift_value_5x5(i, shift_value):
        if i<=4:
            tx = -2*shift_value
        elif i>4 and i<=9:
            tx = -1*shift_value
        elif i>9 and i<=14:
            tx = 0
        elif i>14 and i<=19:
            tx = 1*shift_value
        elif i>19 and i<=24:
            tx = 2*shift_value
        else:
            tx = 3*shift_value
        if i == 0 or (i)%5==0:
            ty = -2*shift_value
        elif i == 1 or (i-1)%5==0:
            ty = -1*shift_value
        elif i == 2 or (i-2)%5==0:
            ty = 0
        elif i == 3 or (i-3)%5==0:
            ty = 1*shift_value
        elif i == 4 or (i-4)%5==0:
            ty = 2*shift_value
        else:
            ty = 3*shift_value
        return tx, ty

    def conv_relu(bottom, ks, nout, stride=1, pad=0):
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                    num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
        relu = L.ReLU(conv, relu_param=dict(negative_slope=0.0), in_place=False)
        return relu

    def block(bottom, ks, nout, dilation=1, stride=1, pad=0):
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                    num_output=nout, pad=pad, dilation=dilation, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
        relu = L.ReLU(conv, relu_param=dict(negative_slope=0.1), in_place=False)
        ccat = L.Concat(relu, bottom, axis=1)
        relu = L.ReLU(ccat, relu_param=dict(negative_slope=0.0), in_place=False)
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
    
    def input_shifting_5x5(bottom):
        con = None
        for i in range(25):
            center = bottom
            tx, ty = shift_value_5x5(i, 1.4) # 0.8
            param_str = json.dumps({'tx': tx, 'ty': ty})
            shift = L.Python(center, module = 'input_shifting_layer', layer = 'InputShiftingLayer', ntop = 1, param_str = param_str)
            if i == 0:
                con = shift
            else:   
                con = L.Concat(*[con, shift], concat_param={'axis': 1})
        return con
    
    def image_data_5x5(batch_size=1):
        con = None
        for i in range(25):
            label, trash = L.ImageData(batch_size=batch_size,
                                    source='/docker/lf_depth/datas/FaceLF/source'+str(i)+'.txt',
                                    transform_param=dict(scale=1./1.),
                                    shuffle=False,
                                    ntop=2,
                                    is_color=False)
            if i == 0:
                con = label
            else:   
                con = L.Concat(*[con, label], concat_param={'axis': 1})
        return con, trash
    
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

    def denseFlowNet_train(batch_size=1):
        n = caffe.NetSpec()

        # Data loading
        n.input, n.trash = L.ImageData(batch_size=batch_size,
                                source='/docker/lf_depth/datas/FaceLF/source'+str(CENTER_SAI)+'.txt',
                                transform_param=dict(scale=1./1.),
                                shuffle=False,
                                ntop=2,
                                is_color=False)
        n.shift = input_shifting_5x5(n.input)
        n.label, n.trash = image_data_5x5(batch_size)

        # Network
        n.conv_relu1 = conv_relu(n.input, 3, 14, 1, 1) # init
        n.block1_add1 = block(n.conv_relu1, 3, 12, 2, 1, 2)
        n.block1_add2 = block(n.block1_add1, 3, 12, 2, 1, 2)
        n.block1_add3 = block(n.block1_add2, 3, 12, 2, 1, 2)
        n.conv_relu2 = conv_relu(n.block1_add3, 3, 50, 1, 1) # trans1
        n.block2_add1 = block(n.conv_relu2, 3, 12, 1, 1, 1)
        n.block2_add2 = block(n.block2_add1, 3, 12, 4, 1, 4)
        n.block2_add3 = block(n.block2_add2, 3, 12, 4, 1, 4)
        n.conv_relu3 = conv_relu(n.block2_add3, 3, 86, 1, 1) # trans2
        n.block3_add1 = block(n.conv_relu3, 3, 12, 8, 1, 8)
        n.block3_add2 = block(n.block3_add1, 3, 12, 8, 1, 8)
        n.block3_add3 = block(n.block3_add2, 3, 12, 8, 1, 8)
        n.conv_relu4 = conv_relu(n.block3_add3, 3, 122, 1, 1) # trans3
        n.block4_add1 = block(n.conv_relu4, 3, 12, 16, 1, 16)
        n.block4_add2 = block(n.block4_add1, 3, 12, 16, 1, 16)
        n.block4_add3 = block(n.block4_add2, 3, 12, 16, 1, 16)
        n.conv_relu5 = conv_relu(n.block4_add3, 3, 158, 1, 1) # trans4
        #n.flow = L.Convolution(n.conv_relu5, kernel_size=9, stride=1,
        #                            num_output=50, pad=4, bias_term=False, weight_filler=dict(type='xavier'))
        #n.flow2 = L.Convolution(n.flow, kernel_size=5, stride=1,
        #                            num_output=50, pad=2, bias_term=False, weight_filler=dict(type='xavier'))
        n.flow4 = L.Convolution(n.conv_relu5, kernel_size=1, stride=1,
                                    num_output=50, pad=0, bias_term=False, weight_filler=dict(type='xavier'))

        # Estimation
        n.flow_h, n.flow_v = L.Slice(n.flow4, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
        n.predict = slice_warp(n.shift, n.flow_h, n.flow_v)
        n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
        n.predict2 = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer', layer = 'BilinearSamplerLayer', ntop = 1)

        # Loss
        n.predict_ver = ver_mean_block(n.predict)
        n.label_ver = ver_mean_block(n.label)
        n.predict_hor = hor_mean_block(n.predict)
        n.label_hor = hor_mean_block(n.label)
        n.loss_ver = L.AbsLoss(n.predict_ver, n.label_ver, loss_weight=1)
        n.loss_hor = L.AbsLoss(n.predict_hor, n.label_hor, loss_weight=1)
        n.loss = L.AbsLoss(n.predict, n.label, loss_weight=1)

        n.trash1 = L.Python(n.flow_h, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                        param_str=str(dict(path='/docker/lf_depth/datas', name='flow_h', mult=30)))
        n.trash2 = L.Python(n.flow_v, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                        param_str=str(dict(path='/docker/lf_depth/datas', name='flow_v', mult=30)))
        n.trash3 = L.Python(n.shift, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                        param_str=str(dict(path='/docker/lf_depth/datas', name='shift', mult=1)))
        n.trash4 = L.Python(n.label, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                        param_str=str(dict(path='/docker/lf_depth/datas', name='label', mult=1)))
        n.trash5 = L.Python(n.predict, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                        param_str=str(dict(path='/docker/lf_depth/datas', name='predict', mult=1)))
        n.trash6 = L.Python(n.predict2, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                        param_str=str(dict(path='/docker/lf_depth/datas', name='predict2', mult=1)))
        return n.to_proto()

    def denseFlowNet_solver(train_net_path, test_net_path=None, snapshot_path=None):
        s = caffe_pb2.SolverParameter()

        s.train_net = train_net_path
        if test_net_path is not None:
            s.test_net.append(test_net_path)
            s.test_interval = 500
            s.test_iter.append(1000)
        else:
            s.test_initialization = False

        s.iter_size = 1
        s.max_iter = 500000

        s.type = 'Adam'
        s.base_lr = 0.0000001 # 0.000005(basic), 

        s.lr_policy = 'fixed'
        s.gamma = 0.75
        s.power = 0.75
        s.stepsize = 1000
        s.momentum = 0.9
        s.momentum2 = 0.999
        #s.weight_decay = 0.0000005
        #s.clip_gradients = 10

        s.display = 1

        s.snapshot = 2000
        if snapshot_path is not None:
            s.snapshot_prefix = snapshot_path

        s.solver_mode = caffe_pb2.SolverParameter.GPU
        return s

    MODEL_PATH = '/docker/lf_depth/models'
    TRAIN_PATH = '/docker/lf_depth/scripts/denseFlowNet_train.prototxt'
    TEST_PATH = '/docker/lf_depth/scripts/denseFlowNet_test.prototxt'
    SOLVER_PATH = '/docker/lf_depth/scripts/denseFlowNet_solver.prototxt'

    def generate_net():
        with open(TRAIN_PATH, 'w') as f:
            f.write(str(denseFlowNet_train(1)))    
        #with open(TEST_PATH, 'w') as f:
        #    f.write(str(denseFlowNet_test(1)))
    
    def generate_solver():
        with open(SOLVER_PATH, 'w') as f:
            f.write(str(denseFlowNet_solver(TRAIN_PATH, None, MODEL_PATH)))

    generate_net()
    generate_solver()
    solver = caffe.get_solver(SOLVER_PATH)
    solver.net.copy_from('/docker/lf_depth/models/denseFlowNet_solver_iter_10000.caffemodel')
    solver.solve()