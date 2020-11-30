# +---+----+----+----+----+----+----+----+----+
# | 0 | 9  | 18 | 27 | 36 | 45 | 54 | 63 | 72 |
# +---+----+----+----+----+----+----+----+----+
# | 1 | 10 | 19 | 28 | 37 | 46 | 55 | 64 | 73 |
# +---+----+----+----+----+----+----+----+----+
# | 2 | 11 | 20 | 29 | 38 | 47 | 56 | 65 | 74 |
# +---+----+----+----+----+----+----+----+----+
# | 3 | 12 | 21 | 30 | 39 | 48 | 57 | 66 | 75 |
# +---+----+----+----+----+----+----+----+----+
# | 4 | 13 | 22 | 31 | 40 | 49 | 58 | 67 | 76 |
# +---+----+----+----+----+----+----+----+----+
# | 5 | 14 | 23 | 32 | 41 | 50 | 59 | 68 | 77 |
# +---+----+----+----+----+----+----+----+----+
# | 6 | 15 | 24 | 33 | 42 | 51 | 60 | 69 | 78 |
# +---+----+----+----+----+----+----+----+----+
# | 7 | 16 | 25 | 34 | 43 | 52 | 61 | 70 | 79 |
# +---+----+----+----+----+----+----+----+----+
# | 8 | 17 | 26 | 35 | 44 | 53 | 62 | 71 | 80 |
# +---+----+----+----+----+----+----+----+----+

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

# +---+----+----+----+----+
# | 0 |  5 | 10 | 15 | 20 |
# +---+----+----+----+----+
# | 1 |  6 | 11 | 16 | 21 |
# +---+----+----+----+----+
# | 2 |  7 | 12 | 17 | 22 |
# +---+----+----+----+----+
# | 3 |  8 | 13 | 18 | 23 |
# +---+----+----+----+----+
# | 4 |  9 | 14 | 19 | 24 |
# +---+----+----+----+----+

import numpy as np
import cv2
import imageio

def trans_order(imgSAIs):
    imgSAIs2 = np.zeros((imgSAIs.shape))
    for i in range(25):
        if i == 0 or (i)%5==0:
            imgSAIs2[:, :, :, i//5] = imgSAIs[:, :, :, i]
        elif i == 1 or (i-1)%5==0:
            imgSAIs2[:, :, :, (i-1)//5+5] = imgSAIs[:, :, :, i]
        elif i == 2 or (i-2)%5==0:
            imgSAIs2[:, :, :, (i-2)//5+5*2] = imgSAIs[:, :, :, i]
        elif i == 3 or (i-3)%5==0:
            imgSAIs2[:, :, :, (i-3)//5+5*3] = imgSAIs[:, :, :, i]
        elif i == 4 or (i-4)%5==0:
            imgSAIs2[:, :, :, (i-4)//5+5*4] = imgSAIs[:, :, :, i]
    return imgSAIs2
    
def make_err_map(pr, gt, path, weight, duration):
    '''
    weight = 1.2
    '''
    sai_amount = pr.shape[-1]
    err = []

    pr = trans_order(pr)
    gt = trans_order(gt)
    pr = trans_order(pr)
    gt = trans_order(gt)
    for i in range(sai_amount):
        pr_tmp = pr[:,:,:,i]
        gt_tmp = gt[:,:,:,i]
        err_tmp = np.abs(pr_tmp-gt_tmp)*weight
        err_tmp = np.uint8(err_tmp)
        err_tmp = cv2.applyColorMap(err_tmp, cv2.COLORMAP_JET)
        err_tmp = cv2.cvtColor(err_tmp, cv2.COLOR_BGR2RGB)
        err.append(err_tmp)

    pr = trans_order(pr)
    gt = trans_order(gt)
    for i in range(sai_amount):
        pr_tmp = pr[:,:,:,i]
        gt_tmp = gt[:,:,:,i]
        err_tmp = np.abs(pr_tmp-gt_tmp)*weight
        err_tmp = np.uint8(err_tmp)
        err_tmp = cv2.applyColorMap(err_tmp, cv2.COLORMAP_JET)
        err_tmp = cv2.cvtColor(err_tmp, cv2.COLOR_BGR2RGB)
        err.append(err_tmp)

    if path != None:
        imageio.mimsave(path, err, duration=duration)

    return err

def make_gird(src, ang_size, path):
    h, w, c, _ = src.shape
    ang_h = ang_size[0]
    ang_w = ang_size[1]
    grid = np.zeros((h*ang_h, w*ang_w, c))

    for ix in range(ang_w):
        for iy in range(ang_h):
            grid[h*iy:h*iy+h, w*ix:w*ix+w, :] = src[:, :, :, ang_w*ix+iy]
    
    if path != None:
        cv2.imwrite(path, grid)

    return grid

def make_gif(src, path, duration):
    src_amount = src.shape[-1]
    src_list = []

    for i in range(src_amount):
        tmp = np.uint8(src[:,:,:,i])
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        src_list.append(tmp)

    imageio.mimsave(path, src_list, duration=duration)

def make_gif2(src, path, duration):
    src_amount = src.shape[-1]
    src_list = []

    for i in range(src_amount):
        tmp = np.uint8(src[:,:,:,i])
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        src_list.append(tmp)

    src = trans_order(src)
    for i in range(src_amount):
        tmp = np.uint8(src[:,:,:,i])
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        src_list.append(tmp)

    imageio.mimsave(path, src_list, duration=duration)

'''
def make_epi(src, ang_size, epi_pt, path)
    h, w, c, _ = src.shape

    sai_mv = np.zeros((sai_h, sai_w, 3, 5, 5))
    sai_epi_ver = np.zeros((sai_h, ang_w*ang_w, 3))
    sai_epi_hor = np.zeros((ang_h*ang_h, sai_w, 3))
    for ax in range(ang_w):
        for ay in range(ang_h):
            sai = sai_img_list[:, :, :, ang_h*ax+ay]
            sai_mv[:, :, :, ay, ax] = sai
            sai_epi_ver[:, ang_w*ax+ay, :] = sai_mv[:, ang_w//2, :, ay, ax]
            sai_epi_hor[ang_h*ax+ay, :, :] = sai_mv[ang_h//2, :, :, ay, ax]
    cv2.imwrite(pr_path+'/sai_epi_ver'+str(i_tot)+'.png', sai_epi_ver)
    cv2.imwrite(pr_path+'/sai_epi_hor'+str(i_tot)+'.png', sai_epi_hor)

    return epi_v, epi_h
'''

'''
    if len(src.shape) == 2:
        h, w = src.shape
        c = 1
    else:
        h, w, c = src.shape
'''

def make_colorwheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def trans_SAIs_to_LF(imgSAIs):
    imgLF = np.zeros((256*1*5, 256*1*5, 3))
    full_LF_crop = np.zeros((256, 256, 3, 5, 5))
    for ax in range(5):
        for ay in range(5):
            full_LF_crop[:, :, :, ax, ay] = imgSAIs[:, :, :, 5*ax+ay]
    for ax in range(5):
        for ay in range(5):
            resized2 = full_LF_crop[:, :, :, ay, ax]
            resized2 = cv2.resize(resized2, dsize=(256*1, 256*1), interpolation=cv2.INTER_CUBIC) 
            imgLF[ay::5, ax::5, :] = resized2
    return imgLF

def trans_order(imgSAIs):
    imgSAIs2 = np.zeros((imgSAIs.shape))
    for i in range(25):
        if i == 0 or (i)%5==0:
            imgSAIs2[:, :, :, i//5] = imgSAIs[:, :, :, i]
        elif i == 1 or (i-1)%5==0:
            imgSAIs2[:, :, :, (i-1)//5+5] = imgSAIs[:, :, :, i]
        elif i == 2 or (i-2)%5==0:
            imgSAIs2[:, :, :, (i-2)//5+5*2] = imgSAIs[:, :, :, i]
        elif i == 3 or (i-3)%5==0:
            imgSAIs2[:, :, :, (i-3)//5+5*3] = imgSAIs[:, :, :, i]
        elif i == 4 or (i-4)%5==0:
            imgSAIs2[:, :, :, (i-4)//5+5*4] = imgSAIs[:, :, :, i]
    return imgSAIs2



def img_to_blob(img):
    if len(img.shape) == 3:
        h, w, c = img.shape
    else:
        h, w = img.shape
        c = 1

    blob = np.zeros((1, c, h, w))

    if c == 3:
        for i in range(3):
            blob[0, i, :, :] = img[:, :, i]
    else:
        blob[0, 0, :, :] = img

    return blob

def blob_to_img(blob):
    b, c, h, w = blob.shape

    if b != 1:
        raise Exception("!!! Batch should be 1 !!!")

    if c == 3:
        img = np.zeros((h, w, c))
        for i in range(3):
            img[:, :, i] = blob[0, i, :, :]
    else:
        img = np.zeros((h, w))
        img = blob[0, 0, :, :] 

    return img

def change_order(imgs, batch_size):
    imgs_result = None
    for j in range(5):
        for i in range(5):
            dum = L.DummyData(shape=dict(dim=[batch_size, 1, 256, 256]))
            temp = L.Crop(imgs, dum, crop_param=dict(axis=1, offset=[5*i+j,0,0]))

            if i==0 and j==0:
                imgs_result = temp
            else:
                imgs_result = L.Concat(*[imgs_result, temp], concat_param={'axis': 1})
    return imgs_result