import numpy as np
import cv2
import sys
import math

np.set_printoptions(threshold=sys.maxsize)

def depth_trans(dep):
    h, w = dep.shape
    dep3 = np.zeros((h, w, 3), np.uint8)

    dep_ch1 = np.uint8(dep // 256)
    dep_ch2 = np.uint8(dep % 256)
    dep_ch3 = np.zeros((h, w), np.uint8)

    dep3[:, :, 0] = dep_ch1
    dep3[:, :, 1] = dep_ch2
    dep3[:, :, 2] = dep_ch3

    return dep3

for i in range(400):
    path = './datas/face_dataset/face_train_dep_1ch/dep_1ch_'+str(i)+'.png'
    dep = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    dep = dep[16:256+16, 16:256+16]

    dep3 = depth_trans(dep)

    path = './datas/face_dataset/face_train_dep_3ch/dep_3ch_'+str(i)+'.png'
    cv2.imwrite(path, dep3)
