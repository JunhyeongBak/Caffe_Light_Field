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

if os.cpu_count() > 2:
    caffe.set_mode_gpu() # For PC
else:
    caffe.set_mode_cpu() # For RK3399

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
    return -tx, -ty

def index_picker_5x5(i):
    
    id_list_9x9_1 = [20, 21, 22, 23, 24,
                    29, 30, 31, 32, 33,
                    38, 39, 40, 41, 42,
                    47, 48, 49, 50, 51,
                    56, 57, 58, 59, 60]
    '''
    id_list_8x8_1 = [9, 10, 11, 12, 13,
                    17, 18, 19, 20, 21,
                    25, 26, 27, 28, 29,
                    33, 34, 35, 36, 37,
                    41, 42, 43, 44, 45]
    '''
    
    return id_list_9x9_1[i]

def image_data_center(batch_size=1, data_path=None, source='train_source', center_id=12):
    center_id = 12
    center, trash = L.ImageData(batch_size=batch_size,
                        source=data_path+'/'+source+str(center_id)+'.txt',
                        transform_param=dict(scale=1./256.),
                        shuffle=False,
                        ntop=2,
                        new_height=192,
                        new_width=192,
                        is_color=False)
    return center, trash

def image_data(batch_size=1, data_path=None, source='train_source', grid_size=25):
    for i in range(grid_size):
        if grid_size == 64:
            i_pick = i
        elif grid_size == 25:
            i_pick = i
        else:
            print('This size is not supported')
        label, trash = L.ImageData(batch_size=batch_size,
                                source=data_path+'/'+source+str(i_pick)+'.txt',
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
            tx, ty = shift_value_5x5(i, SHIFT_VAL)
        elif grid_size == 25:
            i_pick = i
            tx, ty = shift_value_5x5(i_pick, SHIFT_VAL)
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

####################################################################################################
##########                             Fundamental Network                                ##########
####################################################################################################

def denseUNet(input, batch_size):
    def flow_layer(bottom=None, nout=1):
        conv = L.Convolution(bottom, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'), name='flow')
        return conv

    def conv_conv_group_layer(bottom=None, nout=1):
        conv = L.Convolution(bottom, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'))
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.), in_place=True)
        conv = L.Convolution(conv, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                group=nout, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'))
        conv = L.Convolution(conv, num_output=nout, kernel_size=1, stride=1, dilation=1, pad=0,
                                bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'))
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.), in_place=True)
        return conv

    def conv_conv_res_dense_layer(bottom=None, nout=1):
        bottom = L.Convolution(bottom, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'))
        bottom = L.ReLU(bottom, relu_param=dict(negative_slope=0.), in_place=True)
        conv = L.Convolution(bottom, num_output=nout, kernel_size=3, stride=1, dilation=3, pad=3,
                                bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'))
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.), in_place=True)
        conv = L.Convolution(conv, num_output=nout, kernel_size=3, stride=1, dilation=6, pad=6,
                                bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'))
        elth = L.Eltwise(conv, bottom, operation=P.Eltwise.SUM)
        elth = L.ReLU(elth, relu_param=dict(negative_slope=0.), in_place=True)
        return elth   

    def conv_conv_downsample_group_layer(bottom=None, nout=1):
        conv = L.Convolution(bottom, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'))
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.), in_place=True)
        conv = L.Convolution(conv, num_output=nout, kernel_size=3, stride=1, dilation=1, pad=1,
                                group=nout, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'))
        conv = L.Convolution(conv, num_output=nout, kernel_size=1, stride=1, dilation=1, pad=0,
                                bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='msra'))
        conv = L.ReLU(conv, relu_param=dict(negative_slope=0.), in_place=True)
        pool = L.Pooling(conv, kernel_size=3, stride=2, pool=P.Pooling.MAX)
        pool = L.ReLU(pool, relu_param=dict(negative_slope=0.), in_place=True)
        return conv, pool

    def upsample_concat_layer(bottom1=None, bottom2=None, nout=1, crop_size=0):
        deconv = L.Deconvolution(bottom1, convolution_param=dict(num_output=nout, kernel_size=3, stride=2, pad=0))
        deconv = L.ReLU(deconv, relu_param=dict(negative_slope=0.2), in_place=True)
        dum = L.DummyData(shape=dict(dim=[batch_size, nout, crop_size, crop_size]))
        deconv_crop = L.Crop(deconv, dum, crop_param=dict(axis=2, offset=0))
        conc = L.Concat(*[deconv_crop, bottom2], concat_param={'axis': 1})
        return conc

    conv1, poo1 = conv_conv_downsample_group_layer(input, 16)
    conv2, poo2 = conv_conv_downsample_group_layer(poo1, 32)
    conv3, poo3 = conv_conv_downsample_group_layer(poo2, 64)
    conv4, poo4 = conv_conv_downsample_group_layer(poo3, 128)
    
    feature = conv_conv_res_dense_layer(poo4, 256)

    deconv1 = upsample_concat_layer(feature, conv4, 128, 24)
    deconv1 = conv_conv_group_layer(deconv1, 128)
    deconv2 = upsample_concat_layer(deconv1, conv3, 64, 48)
    deconv2 = conv_conv_group_layer(deconv2, 64)
    deconv3 = upsample_concat_layer(deconv2, conv2, 64, 96)
    deconv3 = conv_conv_group_layer(deconv3, 64)
    deconv4 = upsample_concat_layer(deconv3, conv1, 64, 192)
    deconv4 = conv_conv_group_layer(deconv4, 64)

    flow = flow_layer(deconv4, 25*2)

    return flow

####################################################################################################
##########                               Network Generator                                ##########
####################################################################################################

def denseUNet_train(script_path=None, data_path=None, tot=1, sai=25, batch_size=1):
    # Data list txt generating
    for i_sai in range(sai):
        i_pick = index_picker_5x5(i_sai)
        f = open(data_path+'/train_source'+str(i_sai)+'.txt', 'w')
        for i_tot in range(tot):
            data = data_path+'/sai'+str(i_tot)+'_'+str(i_pick)+'.png 0'+'\n'
            f.write(data)
        f.close()

    # Starting Net    
    n = caffe.NetSpec()

    # Data loading
    n.input, n.trash = image_data_center(batch_size, data_path, 'train_source', 12)

    # Fundamental Network
    n.flow25 = denseUNet(n.input, batch_size)   

    # Translation
    n.shift = input_shifting(n.input, batch_size, 25)
    n.flow_h, n.flow_v = L.Slice(n.flow25, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    n.predict = slice_warp(n.shift, n.flow_h, n.flow_v, 25)

    # Loss
    n.label, n.trash_tot = image_data(batch_size, data_path, 'train_source', 25)
    n.dum_predict = L.DummyData(shape=dict(dim=[batch_size, 1, 176, 176]))
    n.predict_crop = L.Crop(n.predict, n.dum_predict, crop_param=dict(axis=2, offset=8))  
    n.dum_label = L.DummyData(shape=dict(dim=[batch_size, 1, 176, 176]))
    n.label_crop = L.Crop(n.label, n.dum_label, crop_param=dict(axis=2, offset=8))

    n.loss = L.AbsLoss(n.predict_crop, n.label_crop)
    n.loss = L.Power(n.loss, power=1.0, scale=1.0, shift=0.0, in_place=True, loss_weight=1)

    #n.trash1 = L.Python(n.flow_h, module='visualization_layer', layer='VisualizationLayer', ntop=1,
    #                param_str=str(dict(path='./datas/face_dataset', name='flow_h', mult=20)))

    # Generating Prototxt
    with open(script_path, 'w') as f:
        f.write(str(n.to_proto()))    

def denseUNet_test(script_path=None, data_path=None, tot=1, sai=25, batch_size=1):
    # Data list txt generating
    for i_sai in range(sai):
        i_pick = index_picker_5x5(i_sai)
        f = open(data_path+'/val_source'+str(i_sai)+'.txt', 'w')
        for i_tot in range(tot):
            data = data_path+'/sai'+str(i_tot)+'_'+str(i_pick)+'.png 0'+'\n'
            f.write(data)
        f.close()

    # Starting Net     
    n = caffe.NetSpec()

    # Data Loading
    n.input, n.trash = image_data_center(batch_size, data_path, 'val_source', 12)

    # Fundamental Network
    n.flow25 = denseUNet(n.input, batch_size)  

    # Translation
    n.shift = input_shifting(n.input, batch_size, 25)
    n.flow_h, n.flow_v = L.Slice(n.flow25, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
    #n.predict = slice_warp(n.shift, n.flow_h, n.flow_v, 25)
    n.flow_con = L.Concat(*[n.flow_v, n.flow_h], concat_param={'axis': 1})
    param_str = json.dumps({'flow_size': 25, 'color_size': 1})
    n.predict = L.Python(*[n.shift, n.flow_con], module = 'bilinear_sampler_layer_3ch', layer = 'BilinearSamplerLayer3ch', ntop = 1, param_str = param_str)

    # Visualization
    n.label, n.trash_tot = image_data(batch_size, data_path, 'val_source', 25)
    n.trash1 = L.Python(n.flow_h, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset', name='flow_h', mult=20)))
    n.trash2 = L.Python(n.flow_v, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset', name='flow_v', mult=20)))
    n.trash3 = L.Python(n.shift, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset', name='input', mult=1*256)))
    n.trash4 = L.Python(n.label, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset', name='label', mult=1*256)))
    n.trash5 = L.Python(n.predict, module='visualization_layer', layer='VisualizationLayer', ntop=1,
                    param_str=str(dict(path='./datas/face_dataset', name='predict', mult=1*256)))

    # Generating Prototxt
    with open(script_path, 'w') as f:
        f.write(str(n.to_proto()))

def denseUNet_deploy(script_path=None, tot=1, sai=25, batch_size=1):
    n = caffe.NetSpec()

    # Data loading
    n.input = L.Input(input_param=dict(shape=dict(dim=[1, 3, 192, 192])))
    n.input_luma = L.Input(input_param=dict(shape=dict(dim=[1, 1, 192, 192])))
    n.input_luma_down = L.Power(n.input_luma, power=1.0, scale=1./256., shift=0, in_place=False)
    n.shift = input_shifting(n.input, batch_size, 25)

    # Network
    n.flow25 = denseUNet(n.input_luma_down, batch_size)  

    # Translation
    n.flow_h, n.flow_v = L.Slice(n.flow25, ntop=2, slice_param=dict(slice_dim=1, slice_point=[25]))
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
        s.test_interval = 25
        s.test_iter.append(1)
    else:
        s.test_initialization = False
    
    s.iter_size = 1
    s.max_iter = 500000
    s.type = 'Adam'
    s.base_lr = 0.0005 # 0.0005,  0.0000001 # 0.000005(basic), 0.0000001
    s.lr_policy = 'step'
    s.gamma = 0.75
    s.power = 0.75
    s.stepsize = 1000
    s.momentum = 0.3
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

def denseUNet_runner(script_path=None, model_path=None, src_color=None):
    # Set a model
    n = caffe.Net(script_path, model_path, caffe.TEST)

    # Input images    
    src_color = cv2.resize(src_color, dsize=(192, 192), interpolation=cv2.INTER_AREA) ###
    src_luma = cv2.cvtColor(src_color, cv2.COLOR_BGR2GRAY)
    src_blob_color = np.zeros((1, 3, 192, 192))
    src_blob_luma = np.zeros((1, 1, 192, 192))
    for i in range(3):
        src_blob_color[:, i, :, :] = src_color[:, :, i]
    src_blob_luma[:, 0, :, :] = src_luma[:, :]
    n.blobs['input'].data[...] = src_blob_color
    n.blobs['input_luma'].data[...] = src_blob_luma

    # Net forward        
    res = n.forward()

    # Get and print sai
    sai = np.zeros((192, 192, 3))
    sai_list =  np.zeros((192, 192, 3, 5*5))
    dst_blob = n.blobs['predict'].data[...]
    for i in range(25):
        dst_blob_slice = dst_blob[:, 3*i:3*i+3, :, :]
        for c in range(3):
            sai[:, :, c] = cv2.resize(dst_blob_slice[0, c, :, :], dsize=(192, 192), interpolation=cv2.INTER_AREA)
        sai_list[:, :, :, i] = sai
        #cv2.imwrite(PATH+'/output/sai'+str(i_tot)+'_'+str(i)+'.png', sai)

    return sai_list

####################################################################################################
##########                                     Main                                       ##########
####################################################################################################

if __name__ == "__main__":
    TRAINSET_PATH = './datas/face_dataset/face_train_9x9_2x'
    TESTSET_PATH = './datas/face_dataset/face_train_9x9_2x'
    DEPLOYSET_PATH = None
    MODEL_PATH = './models'
    TRAIN_PATH = './scripts/denseUNet_train.prototxt'
    TEST_PATH = './scripts/denseUNet_test.prototxt'
    DEPLOY_PATH = './scripts/denseUNet_deploy.prototxt'
    SOLVER_PATH = './scripts/denseUNet_solver.prototxt'

    TOT = 1207
    SAI = 81
    SHIFT_VAL = -0.7*2 #-1.4
    MOD = 'train'
    
    denseUNet_train(script_path=TRAIN_PATH, data_path=TRAINSET_PATH, tot=1207, sai=25, batch_size=1)
    denseUNet_test(script_path=TEST_PATH, data_path=TESTSET_PATH, tot=1207, sai=25, batch_size=1)
    denseUNet_solver(script_train_path=TRAIN_PATH, script_test_path=TEST_PATH, solver_path=SOLVER_PATH, snapshot_path=MODEL_PATH)
    denseUNet_deploy(script_path=DEPLOY_PATH, tot=1207, sai=25, batch_size=1)

    if MOD == 'train': 
        solver = caffe.get_solver(SOLVER_PATH)
        #solver.net.copy_from(MODEL_PATH+'/denseUNet_solver_iter_15000.caffemodel')
        solver.solve()
    elif MOD == 'run':
        test_img = cv2.imread('./test.png', 1)
        test_list = denseUNet_runner(script_path=DEPLOY_PATH, model_path=MODEL_PATH+'/denseUNet_solver_iter_18000.caffemodel', src_color=test_img)
        for i in range(25):
            cv2.imwrite('./sai0_'+str(i)+'.png', test_list[:, :, :, i])

"""
Memo
1. su root
2. ./tools/extra/plot_training_log.py.example 6 /home/junhyeong/docker/Caffe_LF_Syn/plot.png /home/junhyeong/docker/Caffe_LF_Syn/train.log
/opt/caffe/build/tools/caffe train --solver=/docker/Caffe_LF_Syn/scripts/denseUNet_solver.prototxt 2>&1 | tee train.log


"""