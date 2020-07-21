import caffe
import numpy as np
import cv2

class ImageReadLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass
    
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = np.expand_dims(cv2.imread('/docker/lf_depth/datas/FlowerLF/test_online.jpg', cv2.IMREAD_COLOR), axis=0)
        print(top[0].data[...].shape)

    def backward(self, top, propagate_down, bottom):
        # Transfer gradient
        top[0].diff = bottom[0].diff