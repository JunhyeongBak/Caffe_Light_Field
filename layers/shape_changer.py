import caffe
import numpy as np

class ShapeChanger(caffe.Layer):
    def setup(self, bottom, top):
        pass
        
    def reshape(self, bottom, top):
        # caffe's shape : (n, c, h, w)
        # tf's shape : (n, h, w, c)
        self.src = np.asarray(bottom[0].data[...])
        self.dst = np.zeros((self.src.shape[0], self.src.shape[3], self.src.shape[2], self.src.shape[1]))
        top[0].reshape(*self.dst.shape)

    def forward(self, bottom, top):
        self.dst = self.src.reshape(*self.dst.shape)
        top[0].data[...] = self.dst

    def backward(self, bottom, top):
        pass