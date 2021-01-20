import caffe
import numpy as np
import cv2

def depth_trans_3ch(dep):
    h, w = dep.shape
    dep3 = np.zeros((h, w, 3), np.uint8)

    dep_ch1 = np.uint8(dep // 256)
    dep_ch2 = np.uint8(dep % 256)
    dep_ch3 = np.zeros((h, w), np.uint8)

    dep3[:, :, 0] = dep_ch1
    dep3[:, :, 1] = dep_ch2
    dep3[:, :, 2] = dep_ch3

    return dep3

class DepthLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.path = params['path']
    
    def reshape(self, bottom, top):
        top[0].reshape(1)    
     
    def forward(self, bottom, top):
        src = bottom[0].data[...]
        dst = depth_trans_3ch(src[0, 0, :, :])
        cv2.imwrite(self.path, dst)

    def backward(self, top, propagate_down, bottom):
        pass