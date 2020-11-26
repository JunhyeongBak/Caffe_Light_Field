# LF matix
# +---+----+----+----+----+----+----+----+
# | 0 | 8  | 16 | 24 | 32 | 40 | 48 | 56 |
# +---+----+----+----+----+----+----+----+
# | 1 | 9  | 17 | 25 | 33 | 41 | 49 | 57 |
# +---+----+----+----+----+----+----+----+
# | 2 | 10 | 18 | 26 | 34 | 42 | 50 | 58 |
# +---+----+----+----+----+----+----+----+
# | 3 | 11 | 19 | 27 | 35 | 43 | 51 | 59 |
# +---+----+----+----+----+----+----+----+
# | 4 | 12 | 20 | 28 | 36 | 44 | 52 | 60 |
# +---+----+----+----+----+----+----+----+
# | 5 | 13 | 21 | 29 | 37 | 45 | 53 | 61 |
# +---+----+----+----+----+----+----+----+
# | 6 | 14 | 22 | 30 | 38 | 46 | 54 | 62 |
# +---+----+----+----+----+----+----+----+
# | 7 | 15 | 23 | 31 | 39 | 47 | 55 | 63 |
# +---+----+----+----+----+----+----+----+

import caffe
import numpy as np
import cv2
#import imageio

def arduino_map(src, in_min, in_max, out_min, out_max):
    src = np.clip(src,in_min,in_max)
    src = src.astype(dtype=np.float64)
    dst = (src - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    dst = dst.astype(dtype=np.uint8)
    return dst

class VisualizationLayer(caffe.Layer):
    def setup(self, bottom, top):
        # Read params
        params = eval(self.param_str)
        self.path = params['path']
        self.name = params['name']
        self.mult = params['mult']
    
    def reshape(self, bottom, top):
        # Assign top blob shape
        top[0].reshape(1)            

    def forward(self, bottom, top):
        for b in range(bottom[0].data.shape[0]):
            for c in range(bottom[0].data.shape[1]):
                full_name = 'b'+str(b)+'_'+'c'+str(c)+'_'+self.name
                full_path = self.path+'/'+full_name+'.png'
                #print('<'+full_name+'>')
                #print(np.mean(bottom[0].data[b, c, :, :]))

                
                im_color = bottom[0].data[b, c, :, :]
                im_color = arduino_map(im_color, np.min(im_color), np.max(im_color), 255, 0)
                cv2.imwrite(full_path, im_color)

    def backward(self, top, propagate_down, bottom):
        pass