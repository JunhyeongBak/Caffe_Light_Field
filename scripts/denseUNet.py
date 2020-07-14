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
        
    def denseFlowNet_train(batch_size=1):
        n = caffe.NetSpec()

        n.input = L.ImageData(batch_size=batch_size,
                                source='/docker/lf_depth/datas/',
                                transform_param=dict(scale=1./1.),
                                shuffle=False,
                                ntop=1,
                                is_color=False)

        n.label = L.ImageData(batch_size=batch_size,
                                source='/docker/lf_depth/datas/',
                                transform_param=dict(scale=1./1.),
                                shuffle=False,
                                ntop=1,
                                is_color=False)

        mode_str = json.dumps({'id': 0})
        n.shift0 = L.Python(n.input_center, module = 'input_shifting_layer', layer = 'InputShiftingLayer', ntop = 1, param_str = mode_str)

        n.conv_relu1 = conv_relu(n.input_center, 3, 14, 1, 1) # init
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
        n.flow = conv_relu(n.conv_relu5, 3, 158, 2, 1)

        n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
        n.flow_h = L.Power(n.flow_h, power=1.0, scale=-1.0, shift=0.0, in_place=False)
        n.flow_v = L.Power(n.flow_v, power=1.0, scale=-1.0, shift=0.0, in_place=False)

        n.predict0 = L.Warping(n.shift0, n.flow_h, n.flow_v)
        n.loss = L.AbsLoss(n.predict0, n.label0, loss_weight=1)

        bottom_layers = [n.flow_h, n.flow_v, n.left, n.predict, n.right]
        n.print = L.Python(*bottom_layers, module = 'lf_result_layer', layer = 'LfResultLayer', ntop = 1)

        return n.to_proto()

    def denseUNet_solver(train_net_path, test_net_path=None, snapshot_path=None):
        s = caffe_pb2.SolverParameter()

        s.train_net = train_net_path
        if test_net_path is not None:
            s.test_net.append(test_net_path)
            s.test_interval = 500
            s.test_iter.append(1000)
        else:
            s.test_initialization = False

        s.iter_size = 1
        s.max_iter = 5000

        s.type = 'Adam'
        s.base_lr = 0.000001

        s.lr_policy = 'step'
        s.gamma = 0.75
        #s.power = 0.75
        s.stepsize = 2000
        s.momentum = 0.9
        s.weight_decay = 5e-4

        s.display = 100

        s.snapshot = 1000
        if snapshot_path is not None:
            s.snapshot_prefix = snapshot_path

        s.solver_mode = caffe_pb2.SolverParameter.GPU

        return s

    DATA_PATH1 = '/docker/lf_depth/datas/FlowerLF/train_left.txt'
    DATA_PATH2 = '/docker/lf_depth/datas/FlowerLF/train_right.txt'
    MODEL_PATH = '/docker/lf_depth/models'
    TRAIN_PATH = '/docker/lf_depth/scripts/denseUNet_train.prototxt'
    TEST_PATH = '/docker/lf_depth/scripts/denseUNet_test.prototxt'
    SOLVER_PATH = '/docker/lf_depth/scripts/denseUNet_solver.prototxt'

    def generate_net():
        with open(TRAIN_PATH, 'w') as f:
            f.write(str(denseUNet_train(DATA_PATH1, DATA_PATH2, 1)))    
        #with open(TEST_PATH, 'w') as f:
        #    f.write(str(denseUNet_test(DATA_PATH1, DATA_PATH2, 1)))
    
    def generate_solver():
        with open(SOLVER_PATH, 'w') as f:
            f.write(str(denseUNet_solver(TRAIN_PATH, None, MODEL_PATH)))

    generate_net()
    generate_solver()
    solver = caffe.get_solver(SOLVER_PATH)
    solver.solve()