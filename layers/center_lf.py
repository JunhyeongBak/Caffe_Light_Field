import caffe
import numpy as np
import cv2

class CenterLF(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.sai_cnt = params["sai_cnt"]
        
    def reshape(self, bottom, top):
        self.src = np.asarray(bottom[0].data[...])
        self.dst = np.zeros((self.src.shape[0], self.src.shape[1], self.src.shape[2], 1))
        top[0].reshape(*self.dst.shape)

    def forward(self, bottom, top):
        self.dst = np.expand_dims(self.src[:, :, :, self.sai_cnt//2], axis=-1)
        top[0].data[...] = self.dst

    def backward(self, bottom, top):
        pass