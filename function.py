import numpy as np
import cv2
import imageio
import sys

np.set_printoptions(threshold=sys.maxsize)

def shift_value_5x5(i, shift_val):
    if i<=4:
        tx = -2*shift_val
    elif i>4 and i<=9:
        tx = -1*shift_val
    elif i>9 and i<=14:
        tx = 0
    elif i>14 and i<=19:
        tx = 1*shift_val
    elif i>19 and i<=24:
        tx = 2*shift_val
    else:
        tx = 3*shift_val
    if i == 0 or (i)%5==0:
        ty = -2*shift_val
    elif i == 1 or (i-1)%5==0:
        ty = -1*shift_val
    elif i == 2 or (i-2)%5==0:
        ty = 0
    elif i == 3 or (i-3)%5==0:
        ty = 1*shift_val
    elif i == 4 or (i-4)%5==0:
        ty = 2*shift_val
    else:
        ty = 3*shift_val
    return tx, ty

def index_picker_5x5(i, pick_mode='9x9'):
    if pick_mode == '9x9':
        id_list = [20, 21, 22, 23, 24,
                    29, 30, 31, 32, 33,
                    38, 39, 40, 41, 42,
                    47, 48, 49, 50, 51,
                    56, 57, 58, 59, 60]
    elif pick_mode == '8x8':
        id_list = [9, 10, 11, 12, 13,
                    17, 18, 19, 20, 21,
                    25, 26, 27, 28, 29,
                    33, 34, 35, 36, 37,
                    41, 42, 43, 44, 45]
    return id_list[i]

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

def img_list_gen(path, tot_img, tot_sai, pick_mode):
    for i_sai in range(tot_sai):
        i_pick = index_picker_5x5(i_sai, pick_mode)
        f = open(path+'/dataset_list{}.txt'.format(i_sai), 'w')
        for i_tot in range(tot_img):
            f.write(path+'/sai{}_{}.png 0\n'.format(i_tot, i_pick))
        f.close()

    '''
    f = open(path+'/depthset_list.txt', 'w')
    for i_tot in range(tot_img):
        f.write(path+'/sai{}_dep.png 0\n'.format(i_tot))
    f.close()
    '''

'''
    #n.label_depth_org = image_data_depth(batch_size=args.batch_size, data_path=args.trainset_path)

    #n.label_depth_1ch, n.label_depth_2ch, n.depth_trash = L.Slice(n.label_depth_org, ntop=3, slice_param=dict(slice_dim=1, slice_point=[1, 2]))
    #n.label_depth_1ch = L.Power(n.label_depth_1ch, power=1.0, scale=256.0, shift=0.0)
    #n.label_depth = L.Eltwise(n.label_depth_1ch, n.label_depth_2ch, operation=P.Eltwise.SUM)
    #n.label_depth = L.Power(n.label_depth, power=1.0, scale=1/(256.0*256.0), shift=0.0)
    #n.depth_trash = L.Silence(n.depth_trash, ntop=0)
'''

def blob_to_spatial(blob, n_sai):
    SIZE = 256
    c = blob.shape[1]
    
    if c == 3*n_sai:
        spatial = np.zeros((SIZE*5, SIZE*5, 3))
        sais =  np.moveaxis(np.squeeze(blob, 0), 0, -1)

        for ax in range(5):
            for ay in range(5):
                spatial[SIZE*ax:SIZE*ax+SIZE, SIZE*ay:SIZE*ay+SIZE, :] =  sais[:, :, 3*(5*ax+ay):3*(5*ax+ay)+3]
    elif c == n_sai:
        spatial = np.zeros((SIZE*5, SIZE*5))
        sais =  np.moveaxis(np.squeeze(blob, 0), 0, -1)

        for ax in range(5):
            for ay in range(5):
                spatial[SIZE*ax:SIZE*ax+SIZE, SIZE*ay:SIZE*ay+SIZE] =  sais[:, :, (5*ax+ay):(5*ax+ay)+1]
    else:
        raise Exception('This is not a normal image !!!')

    return spatial

def blob_to_angular(blob, n_sai):
    SIZE = 256
    c = blob.shape[1]
    
    if c == 3*n_sai:
        angular = np.zeros((SIZE*5, SIZE*5, 3))
        sais =  np.moveaxis(np.squeeze(blob, 0), 0, -1)

        for ax in range(5):
            for ay in range(5):
                angular[ax::5, ay::5, :] =  sais[:, :, 3*(5*ax+ay):3*(5*ax+ay)+3]
    elif c == n_sai:
        angular = np.zeros((SIZE*5, SIZE*5))
        sais =  np.moveaxis(np.squeeze(blob, 0), 0, -1)

        for ax in range(5):
            for ay in range(5):
                angular[ax::5, ay::5, :] =  sais[:, :, (5*ax+ay):(5*ax+ay)+1]
    else:
        raise Exception('This is not a normal image !!!')

    return angular

def blob_to_img(blob):
    img = np.moveaxis(np.squeeze(blob, 0), 0, -1)

    return img

def img_to_blob(img):
    dim = len(img.shape)

    if dim == 3:
        blob =  np.expand_dims(np.moveaxis(img, -1, 0), 0)
    elif dim == 2:
        blob =  np.expand_dims(np.expand_dims(img, 0), 0)
    else:
        raise Exception('This is not a normal image !!!')

    return blob