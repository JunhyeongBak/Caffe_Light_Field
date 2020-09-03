import caffe
import numpy as np
import cv2

def warp_flow(img_in, flow, color):
    res = np.zeros((img_in.shape))
    for ch in range(3):
        img = img_in[ch, :, :]
        new_flow = np.zeros((flow.shape[1], flow.shape[2], flow.shape[0]), np.float32)
        new_flow[:, :, 0] = flow[1, :, :]
        new_flow[:, :, 1] = flow[0, :, :]
        h, w = new_flow.shape[:2]
        #print('img.shape', img.shape)
        new_flow = new_flow
        new_flow[:,:,0] += np.arange(w)
        new_flow[:,:,1] += np.arange(h)[:,np.newaxis]
        res[ch, :, :] = cv2.remap(img[:, :], new_flow, None, interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_WRAP)

    return res

class BilinearSamplerLayer3ch(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.flow_size = params["flow_size"]
        self.color_size = params["color_size"]
    
    def reshape(self, bottom, top):
        self.src = bottom[0].data[...] # shifted SAIs -> 25 x 3ch
        self.src_shape = bottom[0].data[...].shape 
        self.vec = bottom[1].data[...] # concatenated flows -> 25 x 2ch
        self.vec_shape = bottom[1].data[...].shape 
        top[0].reshape(*self.src_shape)

    def forward(self, bottom, top):
        self.dst = np.zeros((self.src_shape))
        for b in range(self.src_shape[0]):
            for c in range(self.flow_size):
                vec_trans = np.zeros((2, self.src_shape[2], self.src_shape[3]))
                vec_trans[0, :, :] = self.vec[b, c, :, :]
                vec_trans[1, :, :] = self.vec[b, c+self.flow_size, :, :]
                self.dst[b, c*self.color_size:c*self.color_size+self.color_size, :, :] = warp_flow(self.src[b, c*self.color_size:c*self.color_size+self.color_size, :, :], vec_trans, self.color_size)
        top[0].data[...] = self.dst

    def backward(self, top, propagate_down, bottom):
        # Transfer gradient
        top[0].diff = bottom[0].diff