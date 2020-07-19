import caffe
import numpy as np
import cv2

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
"""
def bilinear_sampler(src, v):

    def _get_grid_array(N, H, W, h, w):
        N_i = np.arange(N)
        H_i = np.arange(h+1, h+H+1)
        W_i = np.arange(w+1, w+W+1)
        n, h, w, = np.meshgrid(N_i, H_i, W_i, indexing='ij')
        n = np.expand_dims(n, axis=3) # [N, H, W, 1]
        h = np.expand_dims(h, axis=3) # [N, H, W, 1]
        w = np.expand_dims(w, axis=3) # [N, H, W, 1]
        n = np.cast[np.float32](n) # [N, H, W, 1]
        h = np.cast[np.float32](h) # [N, H, W, 1]
        w = np.cast[np.float32](w) # [N, H, W, 1]

        return n, h, w

    shape = src.shape
    N = shape[0]

    H_ = H = shape[1]
    W_ = W = shape[2]
    h = w = 0

    npad = ((0,0), (1,1), (1,1), (0,0))
    src = np.pad(src, npad, 'constant', constant_values=(0))

    vy, vx = np.split(v, 2, axis=-1)

    n, h, w = _get_grid_array(N, H, W, h, w)

    vx0 = np.floor(vx)
    vy0 = np.floor(vy)
    vx1 = vx0 + 1
    vy1 = vy0 + 1

    H_1 = np.cast[np.float32](H_+1)
    W_1 = np.cast[np.float32](W_+1)

    iy0 = np.clip(vy0 + h, 0., H_1)
    iy1 = np.clip(vy1 + h, 0., H_1)
    ix0 = np.clip(vx0 + w, 0., W_1)
    ix1 = np.clip(vx1 + w, 0., W_1)

    i00 = np.concatenate([n, iy0, ix0], axis=3)
    i01 = np.concatenate([n, iy1, ix0], axis=3)
    i10 = np.concatenate([n, iy0, ix1], axis=3)
    i11 = np.concatenate([n, iy1, ix1], axis=3)

    i00 = np.cast[np.int32](i00)
    i01 = np.cast[np.int32](i01)
    i10 = np.cast[np.int32](i10)
    i11 = np.cast[np.int32](i11)

    idx_shpae = i00.shape
    idx_long = idx_shpae[0] * idx_shpae[1] * idx_shpae[2]

    x00 = []
    i00 = i00.reshape(-1, 3)
    i00_tup = [tuple(x) for x in i00.tolist()]
    for i in range(0, idx_long):
        x00.append(src[i00_tup[i]].tolist())
    x00 = np.asarray(x00)
    x00 = x00.reshape(idx_shpae[0], idx_shpae[1], idx_shpae[2], -1)

    x01 = []
    i01 = i01.reshape(-1, 3)
    i01_tup = [tuple(x) for x in i01.tolist()]
    for i in range(0, idx_long):
        x01.append(src[i01_tup[i]].tolist())
    x01 = np.asarray(x01)
    x01 = x01.reshape(idx_shpae[0], idx_shpae[1], idx_shpae[2], -1)

    x10 = []
    i10 = i10.reshape(-1, 3)
    i10_tup = [tuple(x) for x in i10.tolist()]
    for i in range(0, idx_long):
        x10.append(src[i10_tup[i]].tolist())
    x10 = np.asarray(x10)
    x10 = x10.reshape(idx_shpae[0], idx_shpae[1], idx_shpae[2], -1)

    x11 = []
    i11 = i11.reshape(-1, 3)
    i11_tup = [tuple(x) for x in i11.tolist()]
    for i in range(0, idx_long):
        x11.append(src[i11_tup[i]].tolist())
    x11 = np.asarray(x11)
    x11 = x11.reshape(idx_shpae[0], idx_shpae[1], idx_shpae[2], -1)

    w00 = np.cast[np.float32]((vx1 - vx) * (vy1 - vy))
    w01 = np.cast[np.float32]((vx1 - vx) * (vy - vy0))
    w10 = np.cast[np.float32]((vx - vx0) * (vy1 - vy))
    w11 = np.cast[np.float32]((vx - vx0) * (vy - vy0))
    output1 = np.add(w00*x00, w01*x01)
    output2 = np.add(w10*x10, w11*x11)
    output3 = np.add(output1, output2)

    dst = output3
    return dst
"""
def bilinear_sampler(x, v, resize=False, normalize=False, crop=None, out="CONSTANT"):

    def _get_grid_array(N, H, W, h, w):
        N_i = np.arange(N)
        H_i = np.arange(h+1, h+H+1)
        W_i = np.arange(w+1, w+W+1)
        n, h, w, = np.meshgrid(N_i, H_i, W_i, indexing='ij')
        n = np.expand_dims(n, axis=3) # [N, H, W, 1]
        h = np.expand_dims(h, axis=3) # [N, H, W, 1]
        w = np.expand_dims(w, axis=3) # [N, H, W, 1]
        n = np.cast[np.float32](n) # [N, H, W, 1]
        h = np.cast[np.float32](h) # [N, H, W, 1]
        w = np.cast[np.float32](w) # [N, H, W, 1]

        return n, h, w

    shape = x.shape
    N = shape[0]
    if crop is None:
        H_ = H = shape[1]
        W_ = W = shape[2]
        h = w = 0


    if out == "CONSTANT":
        npad = ((0,0), (1,1), (1,1), (0,0))
        x = np.pad(x, npad, 'constant', constant_values=(0))
        # x = tf.pad(x, ((0,0), (1,1), (1,1), (0,0)), mode='CONSTANT')

    


    vy, vx = np.split(v, 2, axis=3)
    


    n, h, w = _get_grid_array(N, H, W, h, w) # [N, H, W, 3]
    

    
    vx0 = np.floor(vx)
    vy0 = np.floor(vy)
    vx1 = vx0 + 1
    vy1 = vy0 + 1 # [N, H, W, 1]
    


    H_1 = np.cast[np.float32](H_+1)
    W_1 = np.cast[np.float32](W_+1)
    


    iy0 = np.clip(vy0 + h, 0., H_1)
    iy1 = np.clip(vy1 + h, 0., H_1)
    ix0 = np.clip(vx0 + w, 0., W_1)
    ix1 = np.clip(vx1 + w, 0., W_1)
    
    

    i00 = np.concatenate([n, iy0, ix0], axis=3)
    i01 = np.concatenate([n, iy1, ix0], axis=3)
    i10 = np.concatenate([n, iy0, ix1], axis=3)
    i11 = np.concatenate([n, iy1, ix1], axis=3) # [N, H, W, 3]
    


    i00 = np.cast[np.int32](i00)
    i01 = np.cast[np.int32](i01)
    i10 = np.cast[np.int32](i10)
    i11 = np.cast[np.int32](i11)
    

    idx_shpae = i00.shape
    idx_long = idx_shpae[0] * idx_shpae[1] * idx_shpae[2]

    x00 = []
    i00 = i00.reshape(-1, 3)
    i00_tup = [tuple(x) for x in i00.tolist()]
    for i in range(0, idx_long):
        x00.append(x[i00_tup[i]].tolist())
    x00 = np.asarray(x00)
    x00 = x00.reshape(idx_shpae[0], idx_shpae[1], idx_shpae[2], -1)
    
    x01 = []
    i01 = i01.reshape(-1, 3)
    i01_tup = [tuple(x) for x in i01.tolist()]
    for i in range(0, idx_long):
        x01.append(x[i01_tup[i]].tolist())
    x01 = np.asarray(x01)
    x01 = x01.reshape(idx_shpae[0], idx_shpae[1], idx_shpae[2], -1)

    x10 = []
    i10 = i10.reshape(-1, 3)
    i10_tup = [tuple(x) for x in i10.tolist()]
    for i in range(0, idx_long):
        x10.append(x[i10_tup[i]].tolist())
    x10 = np.asarray(x10)
    x10 = x10.reshape(idx_shpae[0], idx_shpae[1], idx_shpae[2], -1)

    x11 = []
    i11 = i11.reshape(-1, 3)
    i11_tup = [tuple(x) for x in i11.tolist()]
    for i in range(0, idx_long):
        x11.append(x[i11_tup[i]].tolist())
    x11 = np.asarray(x11)
    x11 = x11.reshape(idx_shpae[0], idx_shpae[1], idx_shpae[2], -1)


    """ 
    x01 = np.zeros(idx_long)
    i01 = i01.reshape(-1, 3)
    i01_tup = [tuple(x) for x in i01.tolist()]
    for i in range(0, idx_long):
        x01[i] = x[i01_tup[i]]
    x01 = x01.reshape(idx_shpae[0], idx_shpae[1], idx_shpae[2], -1)

    x10 = np.zeros(idx_long)
    i10 = i10.reshape(-1, 3)
    i10_tup = [tuple(x) for x in i10.tolist()]
    for i in range(0, idx_long):
        x10[i] = x[i10_tup[i]]
    x10 = x10.reshape(idx_shpae[0], idx_shpae[1], idx_shpae[2], -1)

    x11 = np.zeros(idx_long)
    i11 = i11.reshape(-1, 3)
    i11_tup = [tuple(x) for x in i11.tolist()]
    for i in range(0, idx_long):
        x11[i] = x[i11_tup[i]]
    x11 = x11.reshape(idx_shpae[0], idx_shpae[1], idx_shpae[2], -1)
    """

    w00 = np.cast[np.float32]((vx1 - vx) * (vy1 - vy))
    w01 = np.cast[np.float32]((vx1 - vx) * (vy - vy0))
    w10 = np.cast[np.float32]((vx - vx0) * (vy1 - vy))
    w11 = np.cast[np.float32]((vx - vx0) * (vy - vy0))
    output1 = np.add(w00*x00, w01*x01)
    output2 = np.add(w10*x10, w11*x11)
    output3 = np.add(output1, output2)

    return output3

"""
class BilinearSamplerLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass
    
    def reshape(self, bottom, top):
        self.src_shape = bottom[0].data[...].shape # lf center sai's shape
        self.src = bottom[0].data[...].reshape(self.src_shape[0], self.src_shape[2], self.src_shape[3], self.src_shape[1]) # caffe shape -> tf shape
        self.vec_shape = bottom[1].data[...].shape # flow vector's shape
        self.vec = bottom[1].data[...].reshape(self.vec_shape[0], self.vec_shape[2], self.vec_shape[3], self.vec_shape[1]) # caffe shape -> tf shape
        top[0].reshape(*self.src_shape) # caffe shape

    def forward(self, bottom, top):
        self.dst = bilinear_sampler(self.src, self.vec)
        cv2.imwrite('/docker/etri/data/FlowerLF/test_input.png', self.src[0, :, :, 0])
        cv2.imwrite('/docker/etri/data/FlowerLF/test_result.png', self.dst[0, :, :, 0])
        top[0].data[...] = self.dst.reshape(*self.src_shape) # caffe shap
        print(self.vec[0, 100:105, 100:105, 0])
        print(self.vec[0, 100:105, 100:105, 1])
        #print('dfasdfadfasdfasdfasdfasdfasdfasdfasdfasd')

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff[...]
        #print('adfadfasdfasdfasdasdfasdfasdfasdfawdfasdsadfasdfasdfasdfasdf', bottom[1].diff[...].shape)
        #print('adfadfasdfasdfasdasdfasdfasdfasdfawdfasdsadfasdfasdfasdfasdf', top[0].diff[...].shape)
        #bottom[1].diff[...] = np.concatenate([top[0].diff[...], top[0].diff[...]], axis=1)
        pass
"""

def warp_flow(img, flow):
    print(flow.shape)
    new_flow = np.zeros((flow.shape[1], flow.shape[2], flow.shape[0]), np.float32)
    new_flow[:, :, 0] = flow[1, :, :]
    new_flow[:, :, 1] = flow[0, :, :]
    h, w = new_flow.shape[:2]
    #print('img.shape', img.shape)
    new_flow = new_flow
    new_flow[:,:,0] += np.arange(w)
    new_flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img[:, :], new_flow, None, interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_WRAP)
    
    return res

class BilinearSamplerLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass
    
    def reshape(self, bottom, top):
        self.src = bottom[0].data[...] # shifted SAIs
        self.src_shape = bottom[0].data[...].shape 
        self.vec = bottom[1].data[...] # concatenated flows
        self.vec_shape = bottom[1].data[...].shape 
        top[0].reshape(*self.src_shape)

    def forward(self, bottom, top):
        self.dst = np.zeros((self.src_shape))
        for b in range(self.src_shape[0]):
            for c in range(self.src_shape[1]):
                vec_trans = np.zeros((2, self.src_shape[2], self.src_shape[3]))
                vec_trans[0, :, :] = self.vec[b, c, :, :]
                vec_trans[1, :, :] = self.vec[b, c+25, :, :]
                self.dst[b, c, :, :] = warp_flow(self.src[b, c, :, :], vec_trans*1.)
        #self.dst[0, :, :, :] = warp_flow(self.src[0, :, :, :], np.concatenate([np.ones((1, 192, 256)) * 20., np.zeros((1, 192, 256)) * 20.], axis=0))
        cv2.imwrite('/docker/lf_depth/datas/FlowerLF/test_input.png', self.src[0, 0, :, :])
        cv2.imwrite('/docker/lf_depth/datas/FlowerLF/test_result.png', self.dst[0, 0, :, :])
        cv2.imwrite('/docker/lf_depth/datas/FlowerLF/ver_flow.png', self.vec[0, 0, :, :] * 15.)
        cv2.imwrite('/docker/lf_depth/datas/FlowerLF/hor_flow.png', self.vec[0, 1, :, :] * 15.)
        #print('Ver flow mean : ', np.mean(self.vec[0, 0, :, :]))
        #print('Hor flow mean : ', np.mean(self.vec[0, 1, :, :]))
        top[0].data[...] = self.dst

    def backward(self, top, propagate_down, bottom):
        # Transfer gradient
        top[0].diff = bottom[0].diff
        #bottom[1].diff[:, 0, :, :] = np.squeeze(top[1].diff, axis=1)
        #bottom[1].diff[:, 1, :, :] = np.squeeze(top[0].diff, axis=1)
        #print('Ver flow diff mean :', np.mean(top[1].diff))
        #print('Hor flow diff mean :', np.mean(top[0].diff))
    