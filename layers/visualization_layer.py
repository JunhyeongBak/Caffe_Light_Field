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

                if self.mult < 256:
                    im_color = abs(bottom[0].data[b, c, :, :]*self.mult)
                    im_color = im_color.astype('uint8')
                    im_color = cv2.applyColorMap(im_color, cv2.COLORMAP_WINTER)
                    cv2.imwrite(full_path, im_color)
                else:
                    im_color = abs(bottom[0].data[b, c, :, :]*self.mult)
                    im_color = im_color.astype('uint8')
                    #im_color = cv2.applyColorMap(im_color, cv2.COLORMAP_WINTER)
                    cv2.imwrite(full_path, im_color)

    def backward(self, top, propagate_down, bottom):
        pass