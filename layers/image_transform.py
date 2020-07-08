import caffe
import numpy as np
import cv2

class ImageTransform(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.tx = params["tx"]
        self.ty = params["ty"]
        
    def reshape(self, bottom, top):
        self.src = np.asarray(bottom[0].data[...])
        print(self.src)
        self.dst = np.zeros((self.src.shape[0], self.src.shape[1], self.src.shape[2], self.src.shape[3]))
        top[0].reshape(*self.dst.shape)

    def forward(self, bottom, top):
        M = np.float32([[1, 0, self.tx], [0, 1, self.ty]])

        for i in range(self.src.shape[0]): # batch
            for j in range(self.src.shape[3]): # ch
                self.dst[i, :, :, j] = cv2.warpAffine(self.src[i, :, :, j], M, (self.src.shape[2], self.src.shape[1]))

        top[0].data[...] = self.dst

    def backward(self, bottom, top):
        pass