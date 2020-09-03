import caffe
import numpy as np
import cv2

def warp_flow(img_in, flow):
    if img_in.shape[0] == 3:
        print(img_in.shape, 1111)
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

    elif img_in.shape[0] == 1:
        print(img_in.shape, 222)
        res = np.zeros((img_in.shape))
        img = img_in[:, :, :]
        new_flow = np.zeros((flow.shape[1], flow.shape[2], flow.shape[0]), np.float32)
        new_flow[:, :, 0] = flow[1, :, :]
        new_flow[:, :, 1] = flow[0, :, :]
        h, w = new_flow.shape[:2]
        #print('img.shape', img.shape)
        new_flow = new_flow
        new_flow[:,:,0] += np.arange(w)
        new_flow[:,:,1] += np.arange(h)[:,np.newaxis]
        print(new_flow.shape)
        res[0, :, :] = cv2.remap(img[0, :, :], new_flow, None, interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_WRAP)

        return res

    else:
        print('brrr.....what you doing??')

class BilinearSamplerLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass
    
    def reshape(self, bottom, top):
        self.src = bottom[0].data[...] # Shifted SAIs -> 64x3 or 25x3 or 64x1 or 25x1 
        self.src_shape = bottom[0].data[...].shape 
        self.vec = bottom[1].data[...] # Concatenated flows -> 64x2 or 25x2
        self.vec_shape = bottom[1].data[...].shape
        self.flow_size =  self.vec_shape[1]//2
        self.color_size = self.src_shape[1]//self.flow_size
        print('self.flow_size', self.flow_size)
        print('self.color_size', self.color_size)
        print('self.src_shape', self.src_shape[1])
        top[0].reshape(*self.src_shape)

    def forward(self, bottom, top):
        self.dst = np.zeros((self.src_shape))
        for i_batch in range(self.src_shape[0]):
            for i_flow in range(self.flow_size):
                vec_trans = np.zeros((2, self.src_shape[2], self.src_shape[3]))
                vec_trans[0, :, :] = self.vec[i_batch, i_flow, :, :]
                vec_trans[1, :, :] = self.vec[i_batch, i_flow+self.flow_size, :, :]
                self.dst[i_batch, i_flow*self.color_size:i_flow*self.color_size+self.color_size, :, :] = warp_flow(self.src[i_batch, i_flow*self.color_size:i_flow*self.color_size+self.color_size, :, :], vec_trans)
        top[0].data[...] = self.dst

    def backward(self, top, propagate_down, bottom):
        #bottom[1].diff = top[0].diff
        for i in range(len(propagate_down)):
            if not propagate_down[i]:
                continue
            sss = np.concatenate((top[0].diff[:], top[0].diff[:]), axis=1)
            bottom[1].diff[...] = sss