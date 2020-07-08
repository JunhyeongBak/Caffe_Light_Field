import caffe
import numpy as np
import cv2
from random import randint

class LfImageDataLayer8x8(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.range = params['range'] # dataset range
        self.batch_size = params['batch_size']
        self.ch = params['ch'] # sai's ch
        self.target_h = params['target_h'] # sai's target height
        self.target_w = params['target_w'] # sai's target width
        
    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size, self.ch, self.target_h, self.target_w) # lf center
        top[1].reshape(self.batch_size, self.ch*64, self.target_h, self.target_w) # lf stack

    def forward(self, bottom, top):
        lf_id = randint(0, self.range-1)
        if self.ch == 1:
            for b_id in range(self.batch_size):
                for sai_id in range(64):
                    sai = cv2.imread('/docker/etri/data/FlowerLF/{}/{}.png'.format(sai_id, lf_id), cv2.IMREAD_GRAYSCALE)
                    sai = cv2.resize(sai, dsize=(self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
                    top[1].data[...][b_id, sai_id*self.ch:sai_id*self.ch+self.ch, :, :] = ((sai/255)-0.5)*2
            for b_id in range(self.batch_size):
                top[0].data[...][b_id, 0:self.ch, :, :] = top[1].data[...][b_id, 27*self.ch:27*self.ch+self.ch, :, :]
        else:
            raise Exception("!!! Not prepared option. !!!")

    def backward(self, top, propagate_down, bottom):
        pass