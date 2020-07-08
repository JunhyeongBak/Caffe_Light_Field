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
import json
from caffe import params as P

np.set_printoptions(threshold=sys.maxsize)
caffe.set_mode_gpu()

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

if __name__ == "__main__":
    def conv_relu(bottom, ks, nout, stride=1, pad=0):
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                    num_output=nout, pad=pad, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
        relu = L.ReLU(conv, in_place=False)
        return relu

    def block(bottom, ks, nout, dilation=1, stride=1, pad=0):
        conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                    num_output=nout, pad=pad, dilation=dilation, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='xavier'))
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
        
    def depthFlowTest_train(source_l, source_r, batch_size=1):
        n = caffe.NetSpec()

        n.left, n.trash = L.ImageData(batch_size=batch_size,
                                source=source_l,
                                transform_param=dict(scale=1./1.),
                                shuffle=False,
                                ntop=2,
                                is_color=False)

        n.right, n.trash = L.ImageData(batch_size=batch_size,
                                source=source_r,
                                transform_param=dict(scale=1./1.),
                                shuffle=False,
                                ntop=2,
                                is_color=False)

        mode_str = json.dumps({'id': 3})
        n.left = L.Python(n.left, module = 'input_shifting_layer', layer = 'InputShiftingLayer', ntop = 1, param_str = mode_str)

        n.conv_relu1 = conv_relu(n.left, 3, 14, 1, 1) # init
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
        n.flow1_conv1 = conv_relu(n.conv_relu5, 3, 158, 1, 1)
        n.flow_h1 = conv_relu(n.flow1_conv1, 3, 18, 1, 1)
        n.flow_h = L.ReLU(n.flow_h1, in_place=False)

        #bottom_layers = [n.left_shift, n.flow]
        #n.predict_v, n.predict_h, n.predict = L.Python(*bottom_layers, module = 'bilinear_sampler_layer', layer = 'BilinearSamplerLayer', ntop = 3)

        n.flow_h = L.Power(n.flow_h, power=1.0, scale=-1.0, shift=0.0, in_place=False)
        n.flow_v = L.Power(n.flow_h, power=1.0, scale=0.0, shift=0.0, in_place=False)

        #bottom_layers = [n.left, n.flow_h]
        #n.result = L.Python(*bottom_layers, module = 'image_print_layer', layer = 'ImagePrintLayer', ntop = 1)

        n.predict = L.Warping(n.left, n.flow_h, n.flow_v)
        n.loss = L.AbsLoss(n.predict, n.right, loss_weight=1)

        bottom_layers = [n.flow_h, n.flow_v, n.left, n.predict, n.right]
        n.print = L.Python(*bottom_layers, module = 'lf_result_layer', layer = 'LfResultLayer', ntop = 1)

        #n.predict_mean_v = L.Python(n.predict_v, module = 'vertical_mean_layer', layer = 'VerticalMeanLayer', ntop = 1)
        #n.label_mean_v = L.Python(n.right, module = 'vertical_mean_layer', layer = 'VerticalMeanLayer', ntop = 1)
        #n.loss_x = L.EuclideanLoss(n.predict_mean_v, n.label_mean_v, loss_weight=1)

        #n.predict_mean_h = L.Python(n.predict_h, module = 'horizontal_mean_layer', layer = 'HorizontalMeanLayer', ntop = 1)
        #n.label_mean_h = L.Python(n.right, module = 'horizontal_mean_layer', layer = 'HorizontalMeanLayer', ntop = 1)
        #n.loss_y = L.EuclideanLoss(n.predict_mean_h, n.label_mean_h, loss_weight=1)

        #n.loss = L.EuclideanLoss(n.predict, n.right, loss_weight=1)

        #bottom_layers = [n.loss_x, n.loss_y, n.loss]
        #n.print = L.Python(*bottom_layers, module = 'print_error_layer', layer = 'PrintErrorLayer', ntop = 1)
        #n.loss_xy = L.Eltwise(n.loss_x, n.loss_y, name='sum', operation=P.El
        # twise.SUM)
        #n.loss_tot = L.Eltwise(n.loss, n.loss_xy, name='sum', operation=P.Eltwise.SUM)

        #bottom_layers = [n.predict, n.right]
        #n.loss = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1, loss_weight=1)
        #bottom_layers = [n.predict, n.right]
        #n.loss = L.Python(*bottom_layers, module = 'euclidean_loss_layer', layer = 'EuclideanLossLayer', ntop = 1, loss_weight=1)
        
        return n.to_proto()

    with open('/docker/lf_depth/scripts/depthFlowTest_train.prototxt', 'w') as f:
        #f.write('force_backward: true\n')
        f.write(str(depthFlowTest_train('/docker/lf_depth/datas/FlowerLF/train_left.txt', '/docker/lf_depth/datas/FlowerLF/train_right.txt', 1)))

    def write_solver(path):
        solver_txt = (# The train/test net protocol buffer definition
                    "train_net: \"/docker/lf_depth/scripts/depthFlowTest_train.prototxt\"\n"
                    #"test_net: \"/docker/etri/denseFlowNet_test.prototxt\"\n"
                
                    # test_iter specifies how many forward passes the test should carry out.
                    # In the case of MNIST, we have test batch size 100 and 100 test iterations,
                    # covering the full 10,000 testing images.
                    #"test_iter: 1\n"

                    # Carry out testing every 500 training iterations.
                    #"test_interval: 5\n"

                    # The base learning rate, momentum and the weight decay of the network.
                    "base_lr: 0.000001\n" # 0.0000000000001
                    "momentum: 0.5\n"
                    #"weight_decay: 0.000000000000005\n"

                    # The learning rate policy
                    "lr_policy: \"fixed\""
                    #"gamma: 0.001\n"
                    #"power: 0.75\n"

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

    #write_solver('/docker/lf_depth/scripts/depthFlowTest_solver.prototxt')

    solver = caffe.get_solver('/docker/lf_depth/scripts/solver_resnet50by2_pooladam_tvl1.prototxt')
    solver.solve()