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
#from datasetPreprocess import load_img, norm_wide, norm_pos, wide_to_pos, pos_to_wide
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
     # write prototxt first! always!!!
    solver = caffe.get_solver('/docker/etri/scripts/solver_resnet50by2_pooladam_tvl1.prototxt')
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