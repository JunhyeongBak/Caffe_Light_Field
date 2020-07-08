import caffe
import numpy as np
import cv2

class ImageTransform3x3(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.shift_val = params["shift_val"]

    def shift_value(self, i):
        if i<=2:
            tx = self.shift_val
        elif i>2 and i<=5:
            tx = 0
        else:
            tx = -1*self.shift_val
        
        if i==0 or (i%3==0 and i>2):
            ty = self.shift_val
        elif i==1 or (i-1)%3==0:
            ty = 0
        else:
            ty = -1*self.shift_val
            
        return tx, ty

    def reshape(self, bottom, top):
        self.src = np.asarray(bottom[0].data[...])
        print(self.src.shape)
        self.dst = np.zeros((self.src.shape[0], self.src.shape[1], self.src.shape[2], self.src.shape[3]*9))
        top[0].reshape(*self.dst.shape)

    def forward(self, bottom, top):
        for i in range(self.src.shape[0]):
            for j in range(self.src.shape[3]*9):
                tx, ty = self.shift_value(j)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                self.dst[i, :, :, j] = cv2.warpAffine(self.src[i, :, :, 0], M, (self.src.shape[2], self.src.shape[1]))
        top[0].data[...] = self.dst

    def backward(self, bottom, top):
        pass