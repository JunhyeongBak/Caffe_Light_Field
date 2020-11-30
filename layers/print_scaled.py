import caffe
import numpy as np
import math
import cv2
import sys
np.set_printoptions(threshold=sys.maxsize) # Numpy maximum print

def arduino_map(src, in_min, in_max, out_min, out_max):
    #src = src_org.astype(dtype=np.float64) 
    dst = (src - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    dst = dst.astype(dtype=np.uint8)
    return dst

class PrintScaledLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.path = params['path']
    
    def reshape(self, bottom, top):
        top[0].reshape(1)    
     
    def forward(self, bottom, top):
        b, c, h, w = bottom[0].data[...].shape
        src = bottom[0].data[...]
        bottom_np = np.zeros((h, w, c, b))
        
        for ic in range(c):
            bottom_np[:, :, ic, 0] = src[0, ic, :, :]
            dst = abs(bottom_np[:, :, :, 0])
            dst_scaled = arduino_map(dst, dst.min(), dst.max(), 0, 255)
            #dst_scaled = dst.astype('uint8')
            cv2.imwrite(self.path, dst_scaled)

            mask_path = (self.path).replace('.png', '_mask.png')
            thr = np.mean(dst_scaled[0:30, 0:30, :])+20
            ret, mask = cv2.threshold(dst_scaled, thr, 255, cv2.THRESH_BINARY)
            cv2.imwrite(mask_path, mask)

    def backward(self, top, propagate_down, bottom):
        pass