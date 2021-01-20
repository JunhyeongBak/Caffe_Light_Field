import caffe
import numpy as np
import cv2
import sys

np.set_printoptions(threshold=sys.maxsize)

class PrintLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass
    
    def reshape(self, bottom, top):
        top[0].reshape(1)    
     
    def forward(self, bottom, top):
        print(bottom[0].data[...])

    def backward(self, top, propagate_down, bottom):
        pass