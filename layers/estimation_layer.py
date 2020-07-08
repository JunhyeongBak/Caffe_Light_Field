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
    i11_tup = [tuple(x) for x in i11.tolist()]/docker/etri/
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

class EstimationLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.shift_val = params['shift_val']
    
    def shift_value(self, i):
        if i<=7:
            tx = 3*self.shift_val
        elif i>7 and i<=15:
            tx = 2*self.shift_val
        elif i>15 and i<=23:
            tx = 1*self.shift_val
        elif i>23 and i<=31:
            tx = 0
        elif i>31 and i<=39:
            tx = -1*self.shift_val
        elif i>39 and i<=47:
            tx = -2*self.shift_val
        elif i>47 and i<=55:
            tx = -3*self.shift_val
        else:
            tx = -4*self.shift_val
        
        if i==0 or (i%8==0 and i>7):
            ty = 3*self.shift_val
        elif i == 1 or (i-1)%8==0:
            ty = 2*self.shift_val
        elif i == 2 or (i-2)%8==0:
            ty = 1*self.shift_val
        elif i == 3 or (i-3)%8==0:
            ty = 0
        elif i == 4 or (i-4)%8==0:
            ty = -1*self.shift_val
        elif i == 5 or (i-5)%8==0:
            ty = -2*self.shift_val
        elif i == 6 or (i-6)%8==0:
            ty = -3*self.shift_val
        else:
            ty = -4*self.shift_val
            
        return tx, ty

    def reshape(self, bottom, top):
        self.src_shape = bottom[0].data[...].shape # lf center sai's shape
        self.src = bottom[0].data[...].reshape(self.src_shape[0], self.src_shape[2], self.src_shape[3], self.src_shape[1]) # caffe shape -> tf shape
        self.vec_shape = bottom[1].data[...].shape # flow vector's shape
        self.vec = bottom[1].data[...].reshape(self.vec_shape[0], self.vec_shape[2], self.vec_shape[3], self.vec_shape[1]) # caffe shape -> tf shape
        top[0].reshape(self.src_shape[0], self.src_shape[1]*64, self.src_shape[2], self.src_shape[3]) # caffe shape

    def forward(self, bottom, top):
        img_shift = np.zeros((self.src_shape[0], self.src_shape[2], self.src_shape[3], self.src_shape[1]))
        trans = np.zeros((self.src_shape[0], self.src_shape[2], self.src_shape[3], self.src_shape[1]))

        for sai_id in range(64):
            tx, ty = self.shift_value(sai_id)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            for b_id in range(self.src_shape[0]):
                for c_id in range(self.src_shape[1]):
                    img_shift[b_id, :, :, c_id] = cv2.warpAffine(self.src[b_id, :, :, c_id], M, (self.src_shape[3], self.src_shape[2]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    #cv2.imwrite('/docker/etri/data/FlowerLF/test_img{}.png'.format(str(sai_id)), ((img_shift[0, :, :, 0]+1)/2)*255)

            if sai_id==0:
                trans = bilinear_sampler(img_shift, self.vec[:, :, :, sai_id*2:(sai_id*2)+2])
            elif sai_id==27:
                trans = np.concatenate([trans, img_shift], axis=-1)     
            else:
                trans = np.concatenate([trans, bilinear_sampler(img_shift, self.vec[:, :, :, sai_id*2:(sai_id*2)+2])], axis=-1)

        for sai_id in range(64):
            top[0].data[...][:, sai_id, :, :] = trans[:, :, :, sai_id]
            #cv2.imwrite('/docker/etri/data/FlowerLF/test_img{}.png'.format(str(sai_id)), ((top[0].data[...][0, sai_id, :, :]+1)/2)*255)
        
        #print(self.vec[0, 96-3:96+2, 128-3:128+2, 0])
        #print(self.vec[0, 96-3:96+2, 128-3:128+2, 1])

    def backward(self, top, propagate_down, bottom):
        #bottom[0].diff[...] = top[0].diff[...]
        #print('adfadfasdfasdfasdasdfasdfasdfasdfawdfasdsadfasdfasdfasdfasdf', bottom[1].diff[...].shape)
        #print('adfadfasdfasdfasdasdfasdfasdfasdfawdfasdsadfasdfasdfasdfasdf', top[0].diff[...].shape)
        bottom[1].diff[...] = np.concatenate([top[0].diff[...], top[0].diff[...]], axis=1)
        pass
