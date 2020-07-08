import caffe
import numpy as np
import cv2

class ImagePrintLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        self.src = bottom[0].data[:]
        top[0].reshape(1)        

    def forward(self, bottom, top):
        print(self.src.shape)
        cv2.imwrite('/docker/lf_depth/flow_test.png', self.src[0, 0, :, :])
        print('fasdfasd', self.src[0, 0, 80-2:80+2, 128-2:128+2])
        top[0].data[...] = 0

    def backward(self, bottom, top):
        bottom[0].diff[...] = top[0].diff[:]