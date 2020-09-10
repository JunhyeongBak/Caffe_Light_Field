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
import skimage
import json
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import math
#from skimage.measure import compare_ssim as ssim

SHIFT_VALUE = 0.4

caffe.set_mode_gpu()
caffe.set_device(0)

def index_5x5_picker(i):
    '''
    list_5x5 = [9, 10, 11, 12, 13,
                17, 18, 19, 20, 21,
                25, 26, 27, 28, 29,
                33, 34, 35, 36, 37,
                41, 42, 43, 44, 45]
    '''
    list_5x5 = [9, 10, 11, 12, 13,
                17, 18, 19, 20, 21,
                25, 26, 27, 28, 29,
                33, 34, 35, 36, 37,
                41, 42, 43, 44, 45]
    i_dst = list_5x5[i]
    return i_dst

def trans_SAIs_to_LF(imgSAIs):
    imgLF = np.zeros((192*1*5, 192*1*5, 3))
    full_LF_crop = np.zeros((192, 192, 3, 5, 5))
    for ax in range(5):
        for ay in range(5):
            full_LF_crop[:, :, :, ax, ay] = imgSAIs[:, :, :, 5*ax+ay]
    for ax in range(5):
        for ay in range(5):
            resized2 = full_LF_crop[:, :, :, ay, ax]
            resized2 = cv2.resize(resized2, dsize=(192*1, 192*1), interpolation=cv2.INTER_LINEAR) 
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

def shift_value_5x5(i):
    if i<=4:
        tx = -2*SHIFT_VALUE
    elif i>4 and i<=9:
        tx = -1*SHIFT_VALUE
    elif i>9 and i<=14:
        tx = 0
    elif i>14 and i<=19:
        tx = 1*SHIFT_VALUE
    elif i>19 and i<=24:
        tx = 2*SHIFT_VALUE
    else:
        tx = 3*SHIFT_VALUE
    if i == 0 or (i)%5==0:
        ty = -2*SHIFT_VALUE
    elif i == 1 or (i-1)%5==0:
        ty = -1*SHIFT_VALUE
    elif i == 2 or (i-2)%5==0:
        ty = 0
    elif i == 3 or (i-3)%5==0:
        ty = 1*SHIFT_VALUE
    elif i == 4 or (i-4)%5==0:
        ty = 2*SHIFT_VALUE
    else:
        ty = 3*SHIFT_VALUE
    return tx, ty

def flow_layer(bottom=None, nout=1):
    conv = L.Convolution(bottom, kernel_size=3, stride=1,
                                num_output=nout, pad=1, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'), name='flow'+str(nout//2)+'0')
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

def input_shifting_5x5(bottom):
    con = None
    for i in range(25):
        center = bottom
        tx, ty = shift_value_5x5(i)
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
    n.shift = input_shifting_5x5(n.input)

    n.input_luma_down = L.Power(n.input_luma, power=1.0, scale=1./256., shift=0, in_place=False)
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

    n.flow = flow_layer(n.conv10, 25*2)

    # Translation
    n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    #n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1)
    param_str = json.dumps({'flow_size': 25, 'color_size': 3})
    n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1, param_str = param_str)

    return n.to_proto()

if __name__ == "__main__":
    TOT = 77
    PATH = './deploy'
    NET_PATH = PATH + '/denseUNet_deploy.prototxt'
    WEIGHTS_PATH = PATH + '/denseUNet_solver_iter_6500.caffemodel'

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
        src_color = cv2.imread(PATH + '/input_flower_8x8/sai'+str(i_tot)+'_27.png', cv2.IMREAD_COLOR)
        if not os.path.isfile(PATH + '/input_flower_8x8/sai'+str(i_tot)+'_27.png'):
            src_color = cv2.imread(PATH + '/input_flower_8x8/sai'+str(i_tot)+'_27.jpg', cv2.IMREAD_COLOR)
            if not os.path.isfile(PATH + '/input_flower_8x8/sai'+str(i_tot)+'_27.jpg'):
                raise Exception('(ERR) The image file is not exist!')
        
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
        sai_list =  np.zeros((192, 192, 3, 5*5))
        dst_blob = net.blobs['predict'].data[...]
        for i in range(25):
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
        sai_uv = np.zeros((192, 192, 3, 5, 5))
        sai_grid = np.zeros((192*5, 192*5, 3))
        sai_epi_ver = np.zeros((192, 5*5, 3))
        sai_epi_hor = np.zeros((5*5, 192, 3))
        for ax in range(5):
            for ay in range(5):
                sai = sai_list[:, :, :, 5*ax+ay]
                sai_uv[:, :, :, ay, ax] = sai
                sai_grid[192*ay:192*ay+192, 192*ax:192*ax+192, :] = sai
                sai_epi_ver[:, 5*ax+ay, :] = sai_uv[:, 192//2, :, ay, ax]
                sai_epi_hor[5*ax+ay, :, :] = sai_uv[192//2, :, :, ay, ax]
        cv2.imwrite(PATH+'/output/sai_grid'+str(i_tot)+'.png', sai_grid)
        cv2.imwrite(PATH+'/output/sai_epi_ver'+str(i_tot)+'.png', sai_epi_ver)
        cv2.imwrite(PATH+'/output/sai_epi_hor'+str(i_tot)+'.png', sai_epi_hor)

        ########## GT ##########  

        # Get sai
        sai_GT_list = np.zeros((192, 192, 3, 25))
        for i in range(25):
            i_change = index_5x5_picker(i)
            sai_GT = cv2.imread(PATH+'/input_flower_8x8/sai'+str(i_tot)+'_'+str(i_change)+'.png')

            sai_GT = cv2.resize(sai_GT, dsize=(192, 192), interpolation=cv2.INTER_AREA)
            sai_GT_list[:, :, :, i] = sai_GT
        
        # Print lf
        sai_GT_list2 = trans_order(sai_GT_list)
        lf_GT = trans_SAIs_to_LF(sai_GT_list2)
        cv2.imwrite(PATH+'/output_GT/result_lf'+str(i_tot)+'.jpg', lf_GT)
        
        # Print grid and epi
        sai_GT_uv = np.zeros((192, 192, 3, 5, 5))
        sai_GT_grid = np.zeros((192*5, 192*5, 3))
        sai_GT_epi_ver = np.zeros((192, 5*5, 3))
        sai_GT_epi_hor = np.zeros((5*5, 192, 3))
        for ax in range(5):
            for ay in range(5):
                sai_GT = sai_GT_list[:, :, :, 5*ax+ay]
                sai_GT_uv[:, :, :, ay, ax] = sai_GT
                sai_GT_grid[192*ay:192*ay+192, 192*ax:192*ax+192, :] = sai_GT
                sai_GT_epi_ver[:, 5*ax+ay, :] = sai_GT_uv[:, 192//2, :, ay, ax]
                sai_GT_epi_hor[5*ax+ay, :, :] = sai_GT_uv[192//2, :, :, ay, ax]
        cv2.imwrite(PATH+'/output_GT/sai_grid'+str(i_tot)+'.png', sai_GT_grid)
        cv2.imwrite(PATH+'/output_GT/sai_epi_ver'+str(i_tot)+'.png', sai_GT_epi_ver)
        cv2.imwrite(PATH+'/output_GT/sai_epi_hor'+str(i_tot)+'.png', sai_GT_epi_hor)

        sai_list = sai_list.astype('uint8')
        sai_GT_list = sai_GT_list.astype('uint8')

        # PSNR
        psnr_tot = 0
        for i in range(25):
            psnr = skimage.metrics.peak_signal_noise_ratio(sai_list[:, :, :, i], sai_GT_list[:, :, :, i])
            psnr_tot = psnr_tot+psnr
        psnr = psnr_tot / 25

        # SSIM
        ssim_noise_tot = 0
        for i in range(25):
            ssim_noise = skimage.metrics.structural_similarity(sai_list[:, :, :, i], sai_GT_list[:, :, :, i], multichannel=True, data_range=sai_list[:, :, :, i].max() - sai_list[:, :, :, i].min())
            ssim_noise_tot = ssim_noise_tot + ssim_noise
        ssim_noise = ssim_noise_tot / 25

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