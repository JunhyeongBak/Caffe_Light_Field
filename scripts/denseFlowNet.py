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

TARGET_SAI = 0
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
# | 9  | 17 | 25 | 33 | 41 |
# +---+----+----+----+----+
# | 10 | 18 | 26 | 34 | 42 |
# +---+----+----+----+----+
# | 11 | 19 | 27 | 35 | 43 |
# +---+----+----+----+----+
# | 12 | 20 | 28 | 36 | 44 |
# +---+----+----+----+----+
# | 13 | 21 | 29 | 37 | 45 |
# +---+----+----+----+----+

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
                con = L.Concat([con, shift], concat_param={'axis': 1})
        return con
        
    def denseFlowNet_train(batch_size=1):
        n = caffe.NetSpec()

        n.input, n.trash = L.ImageData(batch_size=batch_size,
                                source='/docker/lf_depth/datas/FlowerLF/source'+str(CENTER_SAI)+'.txt',
                                transform_param=dict(scale=1./1.),
                                shuffle=False,
                                ntop=2,
                                is_color=False)
        tx, ty = shift_value_5x5(TARGET_SAI, 0.8)
        mode_str = json.dumps({'tx': tx, 'ty': ty})
        n.shift = L.Python(n.input, module = 'input_shifting_layer', layer = 'InputShiftingLayer', ntop = 1, param_str = mode_str)
        
        n.label, n.trash = L.ImageData(batch_size=batch_size,
                                source='/docker/lf_depth/datas/FlowerLF/source'+str(TARGET_SAI)+'.txt',
                                transform_param=dict(scale=1./1.),
                                shuffle=False,
                                ntop=2,
                                is_color=False)

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
        n.flow = conv_relu(n.conv_relu5, 3, 2, 1, 1)

        n.flow_h, n.flow_v = L.Slice(n.flow, ntop=2, slice_param=dict(slice_dim=1, slice_point=[1]))
        n.flow_h = L.Power(n.flow_h, power=1.0, scale=-1.0, shift=0.0, in_place=False)
        n.flow_v = L.Power(n.flow_v, power=1.0, scale=-1.0, shift=0.0, in_place=False)

        n.predict = L.Warping(n.shift, n.flow_h, n.flow_v)
        n.loss = L.AbsLoss(n.predict, n.label, loss_weight=1)

        bottom_layers = [n.flow_h, n.flow_v, n.shift, n.predict, n.label]
        n.print = L.Python(*bottom_layers, module = 'lf_result_layer', layer = 'LfResultLayer', ntop = 1)

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
        s.max_iter = 5000

        s.type = 'Adam'
        s.base_lr = 0.00001

        s.lr_policy = 'fixed'
        s.gamma = 0.75
        s.power = 0.75
        s.stepsize = 2000
        s.momentum = 0.9
        s.momentum2 = 0.999
        #s.weight_decay = 5e-4
        #s.clip_gradients = 10

        s.display = 100

        s.snapshot = 1000
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
    solver.solve()