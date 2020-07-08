import os
import sys
import unittest
import tempfile
import caffe
from caffe import layers as L
import cv2
from caffe.io import caffe_pb2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datasetPreprocess import load_img, norm_wide, norm_pos, wide_to_pos, pos_to_wide
import json

#LF Parameter
ANGULAR_RES_X = 14
ANGULAR_RES_Y = 14
ANGULAR_RES_TARGET = 8*8
IMG_WIDTH = 3584
IMG_HEIGHT = 2688
SPATIAL_HEIGHT = int(IMG_HEIGHT / ANGULAR_RES_Y)
SPATIAL_WIDTH = int(IMG_WIDTH / ANGULAR_RES_X)
CH_INPUT = 1
CH_OUTPUT = 1
SHIFT_VALUE = 0.8

np.set_printoptions(threshold=sys.maxsize)
caffe.set_mode_gpu()

if __name__ == "__main__":
####################################################################################################
#                                      NETWORK GENERATION                                          #
####################################################################################################
    def conv_relu(bottom, ks, nout, stride=1, pad=0):
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                    num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='constant'))
        relu = L.ReLU(conv, in_place=False)
        return relu

    def block(bottom, ks, nout, dilation=1, stride=1, pad=0):
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                    num_output=nout, pad=pad, dilation=dilation, bias_term=False, weight_filler=dict(type='constant'))
        relu = L.ReLU(conv, in_place=False)
        ccat = L.Concat(relu, bottom, axis=1)
        relu = L.ReLU(ccat, in_place=False)
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
        
    def denseFlowNet_train(batch_size=1):
        n = caffe.NetSpec()

        param_str = json.dumps({'range': 2790, 'batch_size': batch_size, 'ch': 1, 'target_h': 192, 'target_w': 256})
        n.data, n.label = L.Python(module = 'lf_image_data_layer8x8', layer = 'LfImageDataLayer8x8', ntop = 2, param_str = param_str)
        #bottom_layers = [n.data, n.label]
        #n.data, n.label = L.Python(*bottom_layers, module = 'lf_result_layer', layer = 'LfResultLayer', ntop = 2)

        n.conv_relu1 = conv_relu(n.data, 3, 14, 1, 1) # init
        n.block1_add1 = block(n.conv_relu1, 3, 12, 2, 1, 2)
        n.block1_add2 = block(n.block1_add1, 3, 12, 2, 1, 2)
        n.block1_add3 = block(n.block1_add2, 3, 12, 2, 1, 2)
        n.conv_relu2 = conv_relu(n.block1_add3, 3, 50, 1, 1) #trans1
        n.block2_add1 = block(n.conv_relu2, 3, 12, 1, 1, 1)
        n.block2_add2 = block(n.block2_add1, 3, 12, 4, 1, 4)
        n.block2_add3 = block(n.block2_add2, 3, 12, 4, 1, 4)
        n.conv_relu3 = conv_relu(n.block2_add3, 3, 86, 1, 1) #trans2
        n.block3_add1 = block(n.conv_relu3, 3, 12, 8, 1, 8)
        n.block3_add2 = block(n.block3_add1, 3, 12, 8, 1, 8)
        n.block3_add3 = block(n.block3_add2, 3, 12, 8, 1, 8)
        n.conv_relu4 = conv_relu(n.block3_add3, 3, 122, 1, 1) #trans3
        n.block4_add1 = block(n.conv_relu4, 3, 12, 16, 1, 16)
        n.block4_add2 = block(n.block4_add1, 3, 12, 16, 1, 16)
        n.block4_add3 = block(n.block4_add2, 3, 12, 16, 1, 16)
        n.conv_relu5 = conv_relu(n.block4_add3, 3, 158, 1, 1) #trans4
        n.flow1_conv1 = conv_relu(n.conv_relu5, 3, 158, 1, 1)
        n.flow = conv_relu(n.flow1_conv1, 3, 64*2, 1, 1)

        bottom_layers = [n.data, n.flow]
        mode_str = json.dumps({'shift_val': 0.8})
        n.predict = L.Python(*bottom_layers, module = 'estimation_layer', layer = 'EstimationLayer', ntop = 1, param_str = mode_str)

        bottom_layers = [n.predict, n.label]
        n.predict, n.label = L.Python(*bottom_layers, module = 'lf_result_layer', layer = 'LfResultLayer', ntop = 2)      

        bottom_layers = [n.predict, n.label]
        n.loss = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1, loss_weight=1)
        #n.predict = L.Python(n.trans, module = 'shape_changer', layer = 'ShapeChanger', ntop = 1)
        #n.ground_truth = L.Python(n.label, module = 'shape_changer', layer = 'ShapeChanger', ntop = 1)
        #bottom_layers = [n.predict, n.ground_truth]
        #n.loss = L.EuclideanLoss(n.predict, n.loss_h)
        
        return n.to_proto()
    
    def denseFlowNet_test(batch_size=1):
        n = caffe.NetSpec()

        param_str = json.dumps({'range': 2790, 'batch_size': batch_size, 'ch': 1, 'target_h': 192, 'target_w': 256})
        n.data, n.label = L.Python(module = 'lf_image_data_layer8x8', layer = 'LfImageDataLayer8x8', ntop = 2, param_str = param_str)
        #bottom_layers = [n.data, n.label]
        #n.data, n.label = L.Python(*bottom_layers, module = 'lf_result_layer', layer = 'LfResultLayer', ntop = 2)

        n.conv_relu1 = conv_relu(n.data, 3, 14, 1, 1) # init
        n.block1_add1 = block(n.conv_relu1, 3, 12, 2, 1, 2)
        n.block1_add2 = block(n.block1_add1, 3, 12, 2, 1, 2)
        n.block1_add3 = block(n.block1_add2, 3, 12, 2, 1, 2)
        n.conv_relu2 = conv_relu(n.block1_add3, 3, 50, 1, 1) #trans1
        n.block2_add1 = block(n.conv_relu2, 3, 12, 1, 1, 1)
        n.block2_add2 = block(n.block2_add1, 3, 12, 4, 1, 4)
        n.block2_add3 = block(n.block2_add2, 3, 12, 4, 1, 4)
        n.conv_relu3 = conv_relu(n.block2_add3, 3, 86, 1, 1) #trans2
        n.block3_add1 = block(n.conv_relu3, 3, 12, 8, 1, 8)
        n.block3_add2 = block(n.block3_add1, 3, 12, 8, 1, 8)
        n.block3_add3 = block(n.block3_add2, 3, 12, 8, 1, 8)# Enter your network definition here.
# Use Shift+Enter to update the visualization.
        n.conv_relu4 = conv_relu(n.block3_add3, 3, 122, 1, 1) #trans3
        n.block4_add1 = block(n.conv_relu4, 3, 12, 16, 1, 16)
        n.block4_add2 = block(n.block4_add1, 3, 12, 16, 1, 16)
        n.block4_add3 = block(n.block4_add2, 3, 12, 16, 1, 16)
        n.conv_relu5 = conv_relu(n.block4_add3, 3, 158, 1, 1) #trans4
        n.flow1_conv1 = conv_relu(n.conv_relu5, 3, 158, 1, 1)
        n.flow = conv_relu(n.flow1_conv1, 3, 64*2, 1, 1)

        bottom_layers = [n.data, n.flow]
        mode_str = json.dumps({'shift_val': 0.8})
        n.predict = L.Python(*bottom_layers, module = 'estimation_layer', layer = 'EstimationLayer', ntop = 1, param_str = mode_str)

        bottom_layers = [n.predict, n.label]
        n.predict, n.label = L.Python(*bottom_layers, module = 'lf_result_layer', layer = 'LfResultLayer', ntop = 2)

        bottom_layers = [n.predict, n.label]
        n.predict_h, n.label_h = L.Python(*bottom_layers, module = 'lf_reloc_layer', layer = 'LfRelocLayer', ntop = 2)

        mode_str = json.dumps({'sais_id': 0})

        n.predict_v0 = L.Python(n.predict, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        n.label_v0 = L.Python(n.label, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        bottom_layers = [n.predict_v0, n.label_v0]
        n.loss_v0 = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)

        n.predict_h0 = L.Python(n.predict_h, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        n.label_h0 = L.Python(n.label_h, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        bottom_layers = [n.predict_h0, n.label_h0]
        n.loss_h0 = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)

        

        mode_str = json.dumps({'sais_id': 1})

        n.predict_v1 = L.Python(n.predict, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        n.label_v1 = L.Python(n.label, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        bottom_layers = [n.predict_v1, n.label_v1]
        n.loss_v1 = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)

        n.predict_h1 = L.Python(n.predict_h, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        n.label_h1 = L.Python(n.label_h, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        bottom_layers = [n.predict_h1, n.label_h1]
        n.loss_h1 = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)



        mode_str = json.dumps({'sais_id': 2})

        n.predict_v2 = L.Python(n.predict, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        n.label_v2 = L.Python(n.label, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        bottom_layers = [n.predict_v2, n.label_v2]
        n.loss_v2 = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)

        n.predict_h2 = L.Python(n.predict_h, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        n.label_h2 = L.Python(n.label_h, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        bottom_layers = [n.predict_h2, n.label_h2]
        n.loss_h2 = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)



        mode_str = json.dumps({'sais_id': 3})

        n.predict_v3 = L.Python(n.predict, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        n.label_v3 = L.Python(n.label, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        bottom_layers = [n.predict_v3, n.label_v3]
        n.loss_v3 = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)

        n.predict_h3 = L.Python(n.predict_h, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        n.label_h3 = L.Python(n.label_h, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        bottom_layers = [n.predict_h3, n.label_h3]
        n.loss_h3 = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)



        mode_str = json.dumps({'sais_id': 4})

        n.predict_v4 = L.Python(n.predict, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        n.label_v4 = L.Python(n.label, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        bottom_layers = [n.predict_v4, n.label_v4]
        n.loss_v4 = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)

        n.predict_h4 = L.Python(n.predict_h, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        n.label_h4 = L.Python(n.label_h, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        bottom_layers = [n.predict_h4, n.label_h4]
        n.loss_h4 = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)



        mode_str = json.dumps({'sais_id': 5})

        n.predict_v5 = L.Python(n.predict, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        n.label_v5 = L.Python(n.label, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        bottom_layers = [n.predict_v5, n.label_v5]
        n.loss_v5 = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)

        n.predict_h5 = L.Python(n.predict_h, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        n.label_h5 = L.Python(n.label_h, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        bottom_layers = [n.predict_h5, n.label_h5]
        n.loss_h5 = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)



        mode_str = json.dumps({'sais_id': 6})

        n.predict_v6 = L.Python(n.predict, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        n.label_v6 = L.Python(n.label, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        bottom_layers = [n.predict_v6, n.label_v6]
        n.loss_v6 = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)

        n.predict_h6 = L.Python(n.predict_h, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        n.label_h6 = L.Python(n.label_h, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        bottom_layers = [n.predict_h6, n.label_h6]
        n.loss_h6 = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)



        mode_str = json.dumps({'sais_id': 7})

        n.predict_v7 = L.Python(n.predict, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        n.label_v7 = L.Python(n.label, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        bottom_layers = [n.predict_v7, n.label_v7]
        n.loss_v7 = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)

        n.predict_h7 = L.Python(n.predict_h, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        n.label_h7 = L.Python(n.label_h, module = 'lf_axis_mean_layer', layer = 'LfAxisMeanLayer', ntop = 1, param_str = mode_str)
        bottom_layers = [n.predict_h7, n.label_h7]
        n.loss_h7 = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)



        n.predict_vt = L.Python(n.predict, module = 'lf_mean_layer', layer = 'LfMeanLayer', ntop = 1)
        n.label_vt = L.Python(n.label, module = 'lf_mean_layer', layer = 'LfMeanLayer', ntop = 1)
        bottom_layers = [n.predict_vt, n.label_vt]
        n.loss_vt = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)

        bottom_layers = [n.predict_h7, n.label_h7]
        n.loss_ht = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)



        bottom_layers = [n.loss_v0, n.loss_v1, n.loss_v2, n.loss_v3, n.loss_v4, n.loss_v5, n.loss_v6, n.loss_v7, n.loss_vt]
        n.loss_v = L.Python(*bottom_layers, module = 'lf_add9_layer', layer = 'LfAdd9Layer', ntop = 1)

        bottom_layers = [n.loss_h0, n.loss_h1, n.loss_h2, n.loss_h3, n.loss_h4, n.loss_h5, n.loss_h6, n.loss_h7, n.loss_ht]
        n.loss_h = L.Python(*bottom_layers, module = 'lf_add9_layer', layer = 'LfAdd9Layer', ntop = 1)
        
        #bottom_layers = [n.loss_v, n.loss_h]
        #n.loss = L.Python(*bottom_layers, module = 'lf_add2_layer', layer = 'LfAdd2Layer', ntop = 1)

        #bottom_layers = [n.predict, n.label]
        #n.loss = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1)
        #n.predict = L.Python(n.trans, module = 'shape_changer', layer = 'ShapeChanger', ntop = 1)
        #n.ground_truth = L.Python(n.label, module = 'shape_changer', layer = 'ShapeChanger', ntop = 1)
        #bottom_layers = [n.predict, n.ground_truth]
        #n.loss = L.EuclideanLoss(n.predict, n.label)
        
        return n.to_proto()

    with open('denseFlowNet_train.prototxt', 'w') as f:
        # f.write('force_backward: true\n')
        f.write(str(denseFlowNet_train(1)))

    #with open('denseFlowNet_test.prototxt', 'w') as f:
    #    f.write(str(denseFlowNet_test(5)))

####################################################################################################
#                                       SOlVER GENERATION                                          #
####################################################################################################
    def write_solver(path):
        solver_txt = (# The train/test net protocol buffer definition
                    "train_net: \"/docker/etri/denseFlowNet_train.prototxt\"\n"
                    #"test_net: \"/docker/etri/denseFlowNet_test.prototxt\"\n"
                
                    # test_iter specifies how many forward passes the test should carry out.
                    # In the case of MNIST, we have test batch size 100 and 100 test iterations,
                    # covering the full 10,000 testing images.
                    #"test_iter: 1\n"

                    # Carry out testing every 500 training iterations.
                    #"test_interval: 5\n"

                    # The base learning rate, momentum and the weight decay of the network.
                    "base_lr: 10\n"
                    "momentum: 0\n"
                    "weight_decay: 0.0005\n"

                    # The learning rate policy
                    "lr_policy: \"inv\""
                    "gamma: 0.001\n"
                    "power: 0.75\n"

                    # Display every 10 iterations
                    "display: 1\n"

                    #The maximum number of iterations
                    "max_iter: 50000\n"

                    #snapshot intermediate results
                    "snapshot: 10000\n"
                    "snapshot_prefix: \"/docker/etri\""
                  )

        with open(path, 'w') as f:
            f.write(solver_txt)

    write_solver('/docker/etri/denseFlowNet_solver.prototxt')
####################################################################################################
#                                      TRAINING & TESTING                                          #
####################################################################################################
     # write prototxt first! always!!!
    solver = caffe.SGDSolver('/docker/etri/denseFlowNet_solver.prototxt')
    solver.solve()


"""
    ## load data
    center_view_a, inLF_a, grid_a =  load_img('/docker/etri/data/FlowerLF/IMG_2220_eslf.png')
    center_view_b, inLF_b, grid_b =  load_img('/docker/etri/data/FlowerLF/IMG_2221_eslf.png')
    center_view_a = np.expand_dims(center_view_a, axis=0)
    center_view_b = np.expand_dims(center_view_b, axis=0)
    inLF_a = np.expand_dims(inLF_a, axis=0)
    inLF_b = np.expand_dims(inLF_b, axis=0)
    center_view = np.concatenate([center_view_a, center_view_b], axis=0)
    inLF = np.concatenate([inLF_a, inLF_b], axis=0)

    ## Input data
    n = caffe.Net('denseFlowNet_train.prototxt', caffe.TRAIN)

    n.blobs['data'].data[...] = center_view
    n.blobs['label'].data[...] = inLF

    n.blobs['vec'].data[...] = v # 마찬가지로 실제로 학습에 의해서 벡터를 만들 수 있다면 필요 없는 부분이다.

    ## Forward propagation
    n.forward()

    ## Read blob
    data_blob = n.blobs['data'].data[...]
    print('data_blob.shape:', data_blob.shape)
    cv2.imwrite('./data/data.png', data_blob[0, :, :, 0])

    label_blob = n.blobs['label'].data[...]
    print('label_blob.shape:', label_blob.shape)
    cv2.imwrite('./data/label0.png', label_blob[0, :, :, 0])
    cv2.imwrite('./data/label1.png', label_blob[0, :, :, 1])
    cv2.imwrite('./data/label2.png', label_blob[0, :, :, 2])
    cv2.imwrite('./data/label3.png', label_blob[0, :, :, 3])
    cv2.imwrite('./data/label4.png', label_blob[0, :, :, 4])
    cv2.imwrite('./data/label5.png', label_blob[0, :, :, 5])
    cv2.imwrite('./data/label6.png', label_blob[0, :, :, 6])
    cv2.imwrite('./data/label7.png', label_blob[0, :, :, 7])
    cv2.imwrite('./data/label8.png', label_blob[0, :, :, 8])
    
    predict_blob = n.blobs['trans'].data[...]
    print('predict_blob.shape:', predict_blob.shape)
    cv2.imwrite('./data/predict0.png', predict_blob[0, :, :, 0])
    cv2.imwrite('./data/predict1.png', predict_blob[0, :, :, 1])
    cv2.imwrite('./data/predict2.png', predict_blob[0, :, :, 2])
    cv2.imwrite('./data/predict3.png', predict_blob[0, :, :, 3])
    cv2.imwrite('./data/predict4.png', predict_blob[0, :, :, 4])
    cv2.imwrite('./data/predict5.png', predict_blob[0, :, :, 5])
    cv2.imwrite('./data/predict6.png', predict_blob[0, :, :, 6])
    cv2.imwrite('./data/predict7.png', predict_blob[0, :, :, 7])
    cv2.imwrite('./data/predict8.png', predict_blob[0, :, :, 8])

    flow1_conv2_shape_blob = n.blobs['flow1_conv2_shape'].data[...]
    print('flow1_conv2_shape_blob.shape:', flow1_conv2_shape_blob.shape)

    loss_blob = n.blobs['loss'].data[...]
    print('loss_blob: ', loss_blob)
"""

"""
vx = np.random.rand(2, 192, 256, 1) * 0.8 # 20
vy = np.random.rand(2, 192, 256, 1) * 0.8 # 20
v0 = np.concatenate([vy, vx], axis=-1)
v = v0

vx = np.random.rand(2, 192, 256, 1) * 0.8 # 20
vy = np.random.rand(2, 192, 256, 1) * 0. # 20
v1 = np.concatenate([vy, vx], axis=-1)
v = np.concatenate([v, v1], axis=-1)

vx = np.random.rand(2, 192, 256, 1) * 0.8 # 20
vy = np.random.rand(2, 192, 256, 1) * -0.8 # 20
v2 = np.concatenate([vy, vx], axis=-1)
v = np.concatenate([v, v2], axis=-1)

vx = np.random.rand(2, 192, 256, 1) * 0. # 20
vy = np.random.rand(2, 192, 256, 1) * 0.8 # 20
v3 = np.concatenate([vy, vx], axis=-1)
v = np.concatenate([v, v3], axis=-1)

vx = np.random.rand(2, 192, 256, 1) * 0. # 20
vy = np.random.rand(2, 192, 256, 1) * 0. # 20
v4 = np.concatenate([vy, vx], axis=-1)
v = np.concatenate([v, v4], axis=-1)

vx = np.random.rand(2, 192, 256, 1) * 0. # 20
vy = np.random.rand(2, 192, 256, 1) * -0.8 # 20
v5 = np.concatenate([vy, vx], axis=-1)
v = np.concatenate([v, v5], axis=-1)

vx = np.random.rand(2, 192, 256, 1) * -0.8 # 20
vy = np.random.rand(2, 192, 256, 1) * 0.8 # 20
v6 = np.concatenate([vy, vx], axis=-1)
v = np.concatenate([v, v6], axis=-1)

vx = np.random.rand(2, 192, 256, 1) * -0.8 # 20
vy = np.random.rand(2, 192, 256, 1) * 0. # 20
v7 = np.concatenate([vy, vx], axis=-1)
v = np.concatenate([v, v7], axis=-1)

vx = np.random.rand(2, 192, 256, 1) * -0.8 # 20
vy = np.random.rand(2, 192, 256, 1) * -0.8 # 20
v8 = np.concatenate([vy, vx], axis=-1)
v = np.concatenate([v, v8], axis=-1) * 0.
"""