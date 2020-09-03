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
import json
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import math
from skimage.measure import compare_ssim as ssim

SHIFT_VALUE = 0.8

caffe.set_mode_gpu()
caffe.set_device(0)

def trans_SAIs_to_LF(imgSAIs):
    imgLF = np.zeros((192*1*8, 192*1*8, 3))
    full_LF_crop = np.zeros((192, 192, 3, 8, 8))
    for ax in range(8):
        for ay in range(8):
            full_LF_crop[:, :, :, ax, ay] = imgSAIs[:, :, :, 8*ax+ay]
    for ax in range(8):
        for ay in range(8):
            resized2 = full_LF_crop[:, :, :, ay, ax]
            resized2 = cv2.resize(resized2, dsize=(192*1, 192*1), interpolation=cv2.INTER_LINEAR) 
            imgLF[ay::8, ax::8, :] = resized2
    return imgLF

def trans_order(imgSAIs):
    imgSAIs2 = np.zeros((imgSAIs.shape))

    for i in range(64):
        if i == 0 or (i)%8==0:
            imgSAIs2[:, :, :, i//8] = imgSAIs[:, :, :, i]
        elif i == 1 or (i-1)%8==0:
            imgSAIs2[:, :, :, (i-1)//8+8] = imgSAIs[:, :, :, i]
        elif i == 2 or (i-2)%8==0:
            imgSAIs2[:, :, :, (i-2)//8+8*2] = imgSAIs[:, :, :, i]
        elif i == 3 or (i-3)%8==0:
            imgSAIs2[:, :, :, (i-3)//8+8*3] = imgSAIs[:, :, :, i]
        elif i == 4 or (i-4)%8==0:
            imgSAIs2[:, :, :, (i-4)//8+8*4] = imgSAIs[:, :, :, i]
        elif i == 5 or (i-5)%8==0:
            imgSAIs2[:, :, :, (i-1)//8+8*5] = imgSAIs[:, :, :, i]
        elif i == 6 or (i-6)%8==0:
            imgSAIs2[:, :, :, (i-2)//8+8*6] = imgSAIs[:, :, :, i]
        elif i == 7 or (i-7)%8==0:
            imgSAIs2[:, :, :, (i-3)//8+8*7] = imgSAIs[:, :, :, i]
    return imgSAIs2

def shift_value_8x8(i):
    if i<=7:
        tx = 3*SHIFT_VALUE
    elif i>7 and i<=15:
        tx = 2*SHIFT_VALUE
    elif i>15 and i<=23:
        tx = 1*SHIFT_VALUE
    elif i>23 and i<=31:
        tx = 0
    elif i>31 and i<=39:
        tx = -1*SHIFT_VALUE
    elif i>39 and i<=47:
        tx = -2*SHIFT_VALUE
    elif i>47 and i<=55:
        tx = -3*SHIFT_VALUE
    else:
        tx = -4*SHIFT_VALUE
    
    if i==0 or (i%8==0 and i>7):
        ty = 3*SHIFT_VALUE
    elif i == 1 or (i-1)%8==0:
        ty = 2*SHIFT_VALUE
    elif i == 2 or (i-2)%8==0:
        ty = 1*SHIFT_VALUE
    elif i == 3 or (i-3)%8==0:
        ty = 0
    elif i == 4 or (i-4)%8==0:
        ty = -1*SHIFT_VALUE
    elif i == 5 or (i-5)%8==0:
        ty = -2*SHIFT_VALUE
    elif i == 6 or (i-6)%8==0:
        ty = -3*SHIFT_VALUE
    else:
        ty = -4*SHIFT_VALUE
        
    return -tx, -ty

def flow_layer(bottom=None, nout=1):
    conv = L.Convolution(bottom, kernel_size=3, stride=1,
                                num_output=nout, pad=1, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'), name='flow64_1')
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=False)
    conv = L.Convolution(conv, kernel_size=3, stride=1,
                                num_output=nout, pad=1, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'), name='flow64_2')
    conv = L.ReLU(conv, relu_param=dict(negative_slope=0.2), in_place=False)
    conv = L.Convolution(conv, kernel_size=1, stride=1,
                                num_output=nout, pad=0, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'), name='flow64_3')
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

def upsample_concat_layer(bottom1=None, bottom2=None, ks=2, nout=1, stride=2, pad=0, batch_size=1, crop_size=0):
    deconv = L.Deconvolution(bottom1, convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride, pad=pad))
    deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=False)
    dum = L.DummyData(shape=dict(dim=[batch_size, nout, crop_size, crop_size]))
    deconv_crop = L.Crop(deconv, dum, crop_param=dict(axis=2, offset=0))
    conc = L.Concat(*[deconv_crop, bottom2], concat_param={'axis': 1})
    return conc

def upsample_concat_crop_layer(bottom1=None, bottom2=None, ks=2, nout=1, stride=2, pad=0, batch_size=1):
    deconv = L.Deconvolution(bottom1, convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride, pad=pad))
    deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=False)
    dum = L.DummyData(shape=dict(dim=[batch_size, 64, 190, 190]))
    deconv_crop = L.Crop(deconv, dum, crop_param=dict(axis=2, offset=0))
    conc = L.Concat(*[deconv_crop, bottom2], concat_param={'axis': 1})
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

def input_shifting_8x8(bottom):
    con = None
    for i in range(64):
        center = bottom
        tx, ty = shift_value_8x8(i)
        param_str = json.dumps({'tx': tx, 'ty': ty})
        shift = L.Python(center, module = 'input_shifting_layer', layer = 'InputShiftingLayer', ntop = 1, param_str = param_str)
        if i == 0:
            con = shift
        else:   
            con = L.Concat(*[con, shift], concat_param={'axis': 1})
    return con

def denseUNet_deploy():
    batch_size = 1
    n = caffe.NetSpec()

    # Input data
    n.input = L.Input(input_param=dict(shape=dict(dim=[1, 3, 192, 192])))
    n.input_luma = L.Input(input_param=dict(shape=dict(dim=[1, 1, 192, 192])))
    n.shift = input_shifting_8x8(n.input)

    n.input_luma_down = L.Power(n.input_luma, power=1.0, scale=1./256., shift=0.0, in_place=False)
    # Network
    n.conv1, n.poo1 = conv_conv_downsample_layer(n.input_luma_down, 3, 16, 2, 1)
    n.conv2, n.poo2 = conv_conv_downsample_layer(n.poo1, 3, 32, 2, 1)
    n.conv3, n.poo3 = conv_conv_downsample_layer(n.poo2, 3, 64, 2, 1)
    n.conv4, n.poo4 = conv_conv_downsample_layer(n.poo3, 3, 128, 2, 1)
    n.conv5, n.poo5 = conv_conv_downsample_layer(n.poo4, 3, 256, 2, 1)

    n.feature = conv_conv_layer(n.poo5, 3, 512, 1, 1)

    n.deconv5 = upsample_concat_layer(n.feature, n.conv5, 3, 256, 2, 0, batch_size, 12)
    n.conv6 = conv_conv_layer(n.deconv5, 3, 256, 1, 1)
    n.deconv6 = upsample_concat_layer(n.conv6, n.conv4, 3, 128, 2, 0, batch_size, 24)
    n.conv7 = conv_conv_layer(n.deconv6, 3, 128, 1, 1)
    n.deconv7 = upsample_concat_layer(n.conv7, n.conv3, 3, 64, 2, 0, batch_size, 48)
    n.conv8 = conv_conv_layer(n.deconv7, 3, 64, 1, 1)
    n.deconv8 = upsample_concat_layer(n.conv8, n.conv2, 3, 64, 2, 0, batch_size, 96)
    n.conv9 = conv_conv_layer(n.deconv8, 3, 64, 1, 1)
    n.deconv9 = upsample_concat_layer(n.conv9, n.conv1, 3, 64, 2, 0, batch_size, 192)
    n.conv10 = conv_conv_layer(n.deconv9, 3, 64, 1, 1)

    n.flow64 = flow_layer(n.conv10, 64*2)

    # Translation
    n.flow_h, n.flow_v = L.Slice(n.flow64, ntop=2, slice_param=dict(slice_dim=1, slice_point=[64]))
    n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    param_str = json.dumps({'flow_size': 64, 'color_size': 3})
    n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1, param_str = param_str)

    return n.to_proto()

if __name__ == "__main__":
    TOT = 77
    PATH = './deploy'
    NET_PATH = PATH + '/denseUNet_deploy.prototxt'
    WEIGHTS_PATH = PATH + '/denseUNet_solver_iter_23000.caffemodel'

    # Generate a network
    def generate_net():
        with open(NET_PATH, 'w') as f:
            f.write(str(denseUNet_deploy()))
    generate_net()

    # Set a model
    net = caffe.Net(NET_PATH, WEIGHTS_PATH, caffe.TEST)

    psnr_mean = 0
    ssim_mean = 0
    time_mean = 0
    for i_tot in range(TOT):
        start = time.time()

        # Input images
        if not os.path.isfile(PATH + '/input_flower_8x8/sai'+str(i_tot)+'_27.png'):
            if not os.path.isfile(PATH + '/input_flower_8x8/sai'+str(i_tot)+'_27.jpg'):
                raise Exception('(ERR) The image file is not exist!')
        src_color = cv2.imread(PATH + '/input_flower_8x8/sai'+str(i_tot)+'_27.png', cv2.IMREAD_COLOR)
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
        sai_list =  np.zeros((192, 192, 3, 8*8))
        dst_blob = net.blobs['predict'].data[...]
        for i in range(64):
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
        sai_uv = np.zeros((192, 192, 3, 8, 8))
        sai_grid = np.zeros((192*8, 192*8, 3))
        sai_epi_ver = np.zeros((192, 8*8, 3))
        sai_epi_hor = np.zeros((8*8, 192, 3))
        for ax in range(8):
            for ay in range(8):
                sai = sai_list[:, :, :, 8*ax+ay]
                sai_uv[:, :, :, ay, ax] = sai
                sai_grid[192*ay:192*ay+192, 192*ax:192*ax+192, :] = sai
                sai_epi_ver[:, 8*ax+ay, :] = sai_uv[:, 192//2, :, ay, ax]
                sai_epi_hor[8*ax+ay, :, :] = sai_uv[192//2, :, :, ay, ax]
        cv2.imwrite(PATH+'/output/sai_grid'+str(i_tot)+'.png', sai_grid)
        cv2.imwrite(PATH+'/output/sai_epi_ver'+str(i_tot)+'.png', sai_epi_ver)
        cv2.imwrite(PATH+'/output/sai_epi_hor'+str(i_tot)+'.png', sai_epi_hor)

        ########## GT ##########  

        # Get sai
        sai_GT_list = np.zeros((192, 192, 3, 64))
        for i in range(64):
            sai_GT = cv2.imread(PATH+'/input_flower_8x8/sai'+str(i_tot)+'_'+str(i)+'.png')
            sai_GT = cv2.resize(sai_GT, dsize=(192, 192), interpolation=cv2.INTER_AREA)
            sai_GT_list[:, :, :, i] = sai_GT
        
        # Print lf
        sai_GT_list2 = trans_order(sai_GT_list)
        lf_GT = trans_SAIs_to_LF(sai_GT_list2)
        cv2.imwrite(PATH+'/output_GT/result_lf'+str(i_tot)+'.jpg', lf_GT)
        
        # Print grid and epi
        sai_GT_uv = np.zeros((192, 192, 3, 8, 8))
        sai_GT_grid = np.zeros((192*8, 192*8, 3))
        sai_GT_epi_ver = np.zeros((192, 8*8, 3))
        sai_GT_epi_hor = np.zeros((8*8, 192, 3))
        for ax in range(8):
            for ay in range(8):
                sai_GT = sai_GT_list[:, :, :, 5*ax+ay]
                sai_GT_uv[:, :, :, ay, ax] = sai_GT
                sai_GT_grid[192*ay:192*ay+192, 192*ax:192*ax+192, :] = sai_GT
                sai_GT_epi_ver[:, 8*ax+ay, :] = sai_GT_uv[:, 192//2, :, ay, ax]
                sai_GT_epi_hor[8*ax+ay, :, :] = sai_GT_uv[192//2, :, :, ay, ax]
        cv2.imwrite(PATH+'/output_GT/sai_grid'+str(i_tot)+'.png', sai_GT_grid)
        cv2.imwrite(PATH+'/output_GT/sai_epi_ver'+str(i_tot)+'.png', sai_GT_epi_ver)
        cv2.imwrite(PATH+'/output_GT/sai_epi_hor'+str(i_tot)+'.png', sai_GT_epi_hor)

        # PSNR
        psnr_tot = 0
        for i in range(64):
            mse = np.mean( (sai_list[:, :, :, i] - sai_GT_list[:, :, :, i]) ** 2 )
            psnr = 0
            if mse == 0:
                psnr = 100
            else:
                PIXEL_MAX = 255.0
                psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
            psnr_tot = psnr_tot+psnr
        psnr = psnr_tot / 64

        # SSIM
        ssim_noise_tot = 0
        for i in range(64):
            ssim_noise = ssim(sai_list[:, :, :, i], sai_GT_list[:, :, :, i], multichannel=True, data_range=sai_grid.max() - sai_grid.min())
            ssim_noise_tot = ssim_noise_tot + ssim_noise
        ssim_noise = ssim_noise_tot / 64

        end = time.time()

        psnr_mean = psnr_mean + psnr
        ssim_mean = ssim_mean + ssim_noise
        time_mean = time_mean + end-start
        print('i : '+str(i_tot)+' | '+'time labs : '+str(end-start)+' | '+'PSNR : '+str(psnr)+' | '+'SSIM : '+str(ssim_noise))

    psnr_mean = psnr_mean / TOT
    ssim_mean = ssim_mean / TOT
    time_mean = time_mean / TOT
    print('Total result | '+'time labs : '+str(time_mean)+' | '+'PSNR : '+str(psnr_mean)+' | '+'SSIM : '+str(ssim_mean))
    exit(0)