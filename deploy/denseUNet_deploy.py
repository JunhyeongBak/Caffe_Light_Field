import os
import cv2
import time
import numpy as np
import argparse
import caffe
import json
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

caffe.set_mode_cpu()
caffe.set_device(0)

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

def denseUNet_deploy():
    n = caffe.NetSpec()

    # Input data
    n.input = L.Input(input_param=dict(shape=dict(dim=[1, 3, 192, 256])))
    n.luma = luma_layer(n.input)
    n.shift = input_shifting_5x5(n.input)

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

    n.predict0, n.remain = L.Slice(n.predict, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict1, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict2, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict3, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict4, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict5, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict6, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict7, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict8, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict9, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict10, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict11, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict12, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict13, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict14, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict15, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict16, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict17, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict18, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict19, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict20, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict21, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict22, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict23, n.remain = L.Slice(n.remain, ntop=2, slice_param=dict(slice_dim=1, slice_point=[3]))
    n.predict24 = n.remain
    return n.to_proto()

if __name__ == "__main__":
    start = time.time()
    
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('-m', '--mode', type=str, required=False, help='Select mode')
    parser.add_argument('-p', '--path', type=str, required=False, help='Write path')
    args = parser.parse_args()
    MODEL_PATH = args.path + '/denseUNet_deploy.prototxt'
    WEIGHTS_PATH = args.path + '/denseUNet_deploy.caffemodel'
    SRC_PATH = args.path + '/input_image.jpg'
    DST_PATH =  args.path
    # Ex) python3 denseUNet_deploy.py -m run -p /docker/lf_depth/deploy

    if args.mode == 'run':
        net = caffe.Net(MODEL_PATH, WEIGHTS_PATH, caffe.TEST)
        src = cv2.imread(SRC_PATH, cv2.IMREAD_COLOR)
        src_blob = np.zeros((1, 3, 192, 256))
        for i in range(3):
            src_blob[0, i, :, :] = src[:, :, i]
        net.blobs['input'].data[...] = src_blob
        
        res = net.forward()

        dst = np.zeros((192, 256, 3))
        for i in range(25):
            dst_blob = net.blobs['predict'+str(i)].data[...]
            for c in range(3):
                dst[:, :, c] = dst_blob[0, c, :, :]
            cv2.imwrite(DST_PATH+'/result_image'+str(i)+'.png', dst)
    elif args.mode == 'gen':
        def generate_net():
            with open(MODEL_PATH, 'w') as f:
                f.write(str(denseUNet_deploy()))
        generate_net()
    elif args.mode == 'gt':
        pass
    else:
        pass

    end = time.time()
    print('time labs : ', str(end-start))