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

if __name__ == "__main__":
    def shift_value_5x5(i, shift_value):
        if i<=4:
            tx = 2*shift_value
        elif i>4 and i<=9:
            tx = 1*shift_value
        elif i>9 and i<=14:
            tx = 0
        elif i>14 and i<=19:
            tx = -1*shift_value
        elif i>19 and i<=24:
            tx = -2*shift_value
        else:
            tx = -3*shift_value
        if i == 0 or (i)%5==0:
            ty = 2*shift_value
        elif i == 1 or (i-1)%5==0:
            ty = 1*shift_value
        elif i == 2 or (i-2)%5==0:
            ty = 0
        elif i == 3 or (i-3)%5==0:
            ty = -1*shift_value
        elif i == 4 or (i-4)%5==0:
            ty = -2*shift_value
        else:
            ty = -3*shift_value
        return -tx, -ty

    def flow_layer(bottom=None, nout=1):
        conv = L.Convolution(bottom, kernel_size=3, stride=1,
                                    num_output=nout, pad=1, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=False)
        conv = L.Convolution(conv, kernel_size=3, stride=1,
                                    num_output=nout, pad=1, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=False)
        conv = L.Convolution(conv, kernel_size=1, stride=1,
                                    num_output=nout, pad=0, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
        return conv
    
    def conv_layer(bottom=None, ks=3, nout=1, stride=1, pad=1):
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                    num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=False)
        return conv
    
    def conv_final_layer(bottom=None, ks=3, nout=1, stride=1, pad=1):
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                    num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
        conv = L.TanH(conv, in_place=False)
        return conv

    def conv_conv_layer(bottom=None, ks=3, nout=1, stride=1, pad=1):
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                    num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=False)
        conv = L.Convolution(conv, kernel_size=ks, stride=stride,
                                    num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=False)
        return conv
    
    def downsample_layer(bottom=None, ks=3, stride=2):
        pool = L.Pooling(bottom, kernel_size=ks, stride=stride, pool=P.Pooling.MAX)
        pool = L.ReLU(pool, relu_param=dict(negative_slope=0.2), in_place=False)
        return pool

    def upsample_layer(bottom=None, ks=2, nout=1, stride=2):
        deconv = L.Deconvolution(bottom, convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride, pad=0))
        deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=False)
        return deconv

    def upsample_concat_layer(bottom1=None, bottom2=None, ks=2, nout=1, stride=2):
        deconv = L.Deconvolution(bottom1, convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride, pad=0))
        deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=False)
        conc = L.Concat(*[deconv, bottom2], concat_param={'axis': 1})
        return conc

    def upsample_conv_conv_layer(bottom=None, ks=3, nout=1, stride=2, pad=1):
        deconv = L.Deconvolution(bottom, convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride))
        deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=False)
        conv = L.Convolution(deconv, kernel_size=ks, stride=1,
                                    num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=False)
        conv = L.Convolution(conv, kernel_size=ks, stride=1,
                                    num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
        conv = L.TanH(conv, in_place=False)
        return conv

    def conv_conv_downsample_layer(bottom=None, ks=3, nout=1, stride=2, pad=1):
        conv = L.Convolution(bottom, kernel_size=ks, stride=1,
                                    num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=False)
        conv = L.Convolution(conv, kernel_size=ks, stride=1,
                                    num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=False)
        pool = L.Pooling(conv, kernel_size=ks, stride=stride, pool=P.Pooling.MAX)
        pool = L.ReLU(pool, relu_param=dict(negative_slope=0.2), in_place=False)
        return conv, pool

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
            tx, ty = shift_value_5x5(i, 0.8)
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
                                    source='/docker/lf_depth/datas/FlowerLF/source'+str(i)+'.txt',
                                    transform_param=dict(scale=1./1.),
                                    shuffle=False,
                                    ntop=2,
                                    is_color=False)
            if i == 0:
                con = label
            else:   
                con = L.Concat(*[con, label], concat_param={'axis': 1})
        return con, trash
        
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

    def luma_layer(bottom):
        b, g, r = L.Slice(bottom, ntop=3, slice_param=dict(slice_dim=1, slice_point=[1,2]))
        y_r = L.Power(r, power=1.0, scale=0.299, shift=0.0, in_place=False)
        y_g = L.Power(g, power=1.0, scale=0.587, shift=0.0, in_place=False)
        y_b = L.Power(b, power=1.0, scale=0.114, shift=0.0, in_place=False)
        y = L.Eltwise(y_r, y_g, operation=P.Eltwise.SUM)
        y = L.Eltwise(y, y_b, operation=P.Eltwise.SUM)        
        return y

    def denseUNet_demo(batch_size=1):
        n = caffe.NetSpec()

        # Data loading
        n.input, n.trash = L.ImageData(batch_size=batch_size,
                                source='/docker/lf_depth/datas/FlowerLF/input_image.txt',
                                transform_param=dict(scale=1./1.),
                                shuffle=False,
                                ntop=2,
                                is_color=True)
        n.luma = luma_layer(n.input)
        n.shift = input_shifting_5x5(n.input)

        #n.shape1 = L.Python(n.luma, module='print_shape_layer', layer='PrintShapeLayer', ntop=1, param_str=str(dict(comment='n.luma')))

        # Network
        n.conv1, n.poo1 = conv_conv_downsample_layer(n.luma, 3, 16, 2, 1)
        n.conv2, n.poo2 = conv_conv_downsample_layer(n.poo1, 3, 32, 2, 1)
        n.conv3, n.poo3 = conv_conv_downsample_layer(n.poo2, 3, 64, 2, 1)
        n.conv4, n.poo4 = conv_conv_downsample_layer(n.poo3, 3, 128, 2, 1)
        n.conv5, n.poo5 = conv_conv_downsample_layer(n.poo4, 3, 256, 2, 1)

        n.feature = conv_conv_layer(n.poo5, 3, 512, 1, 1)
        
        n.deconv5 = upsample_concat_layer(n.feature, n.conv5, 2, 256, 2)
        n.conv6 = conv_conv_layer(n.deconv5, 3, 256, 1, 1)
        n.deconv6 = upsample_concat_layer(n.conv6, n.conv4, 2, 128, 2)
        n.conv7 = conv_conv_layer(n.deconv6, 3, 128, 1, 1)
        n.deconv7 = upsample_concat_layer(n.conv7, n.conv3, 2, 64, 2)
        n.conv8 = conv_conv_layer(n.deconv7, 3, 64, 1, 1)
        n.deconv8 = upsample_concat_layer(n.conv8, n.conv2, 2, 64, 2)
        n.conv9 = conv_conv_layer(n.deconv8, 3, 64, 1, 1)
        n.deconv9 = upsample_concat_layer(n.conv9, n.conv1, 2, 64, 2)
        n.conv10 = conv_conv_layer(n.deconv9, 3, 64, 1, 1)

        n.flow4 = flow_layer(n.conv10, 25*2)

        # Translation
        n.flow_h, n.flow_v = L.Slice(n.flow4, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
        n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
        n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1)

        # Visualization
        
        n.trash1 = L.Python(n.flow_h, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                        param_str=str(dict(path='/docker/lf_depth/datas', name='demo_flow_h', mult=30)))
        n.trash2 = L.Python(n.flow_v, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                        param_str=str(dict(path='/docker/lf_depth/datas', name='demo_flow_v', mult=30)))
        """
        n.trash3 = L.Python(n.shift, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                        param_str=str(dict(path='/docker/lf_depth/datas', name='demo_shift', mult=1)))
        n.trash4 = L.Python(n.label, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                        param_str=str(dict(path='/docker/lf_depth/datas', name='demo_label', mult=1)))
        """
        n.trash5 = L.Python(n.predict, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                        param_str=str(dict(path='/docker/lf_depth/datas', name='demo_predict', mult=1)))
        
        return n.to_proto()

    def denseUNet_demo_solver(demo_net_path):
        s = caffe_pb2.SolverParameter()

        s.train_net = demo_net_path

        s.iter_size = 1
        s.max_iter = 500000

        s.type = 'Adam'
        s.base_lr = 0.

        s.lr_policy = 'fixed'
        s.gamma = 0.75
        s.power = 0.75
        s.stepsize = 1000
        s.momentum = 0.9
        s.momentum2 = 0.999
        #s.weight_decay = 0.0000005
        #s.clip_gradients = 10

        s.display = 1
        s.solver_mode = caffe_pb2.SolverParameter.GPU

        return s

    MODEL_PATH = '/docker/lf_depth/models/denseUNet.caffemodel'
    DEMO_PATH = '/docker/lf_depth/scripts/denseUNet_demo.prototxt'
    SOLVER_PATH = '/docker/lf_depth/scripts/denseUNet_demo_solver.prototxt'

    def generate_net():
        with open(DEMO_PATH, 'w') as f:
            f.write(str(denseUNet_demo(1)))
    
    def generate_solver():
        with open(SOLVER_PATH, 'w') as f:
            f.write(str(denseUNet_demo_solver(DEMO_PATH)))

    generate_net()
    generate_solver()
    solver = caffe.get_solver(SOLVER_PATH)
    solver.net.copy_from(MODEL_PATH)
    solver.solve()