import caffe
import numpy as np
import cv2

class BGRtoYUV(caffe.Layer):

    def setup(self, bottom, top):
        pass
        
    def reshape(self, bottom, top):
        ### blob to numpy image ###
        # blob's shape : (n, c, h, w)
        # images's shape : (n, h, w, c)
        # image's shape : (h, w, c)
        self.src = np.asarray(bottom[0].data[...])
        self.shape = self.src.shape
        top[0].reshape(*self.src.shape)
        self.src = self.src.reshape(self.shape[0], self.shape[2], self.shape[3], self.shape[1])
        self.dst = np.zeros((self.shape[0], self.shape[2], self.shape[3], self.shape[1]))

    def forward(self, bottom, top):
        # self.src[:, 0, :, :] Blue
        # self.src[:, 1, :, :] Green
        # self.src[:, 2, :, :] Red
        # self.dst[:, 0, :, :] Y
        # self.dst[:, 1, :, :] U
        # self.dst[:, 2, :, :] V

        self.dst[:, :, :, 0] = (0.257*(self.src[:, :, :, 2])+(0.504*self.src[:, :, :, 1])+(0.095*self.src[:, :, :, 0])+16)
        self.dst[:, :, :, 1] = (-0.148*(self.src[:, :, :, 2])+(-0.291*self.src[:, :, :, 1])+(0.499*self.src[:, :, :, 0])+128)
        self.dst[:, :, :, 2] = (0.439*(self.src[:, :, :, 2])+(-0.368*self.src[:, :, :, 1])+(-0.071*self.src[:, :, :, 0])+128)

        self.dst = self.dst.reshape(self.shape[0], self.shape[1], self.shape[2], self.shape[3])
        top[0].data[...] = self.dst
        print('adfsadfd', self.dst.shape)

    def backward(self, bottom, top):
        pass