import os
import numpy as np
import cv2
import imageio
import sys
import math

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

# raise Exception('This is not a normal image !!!')

def input_shifting(img, shift_val):
    lf = np.zeros((256, 256, 25), np.float32)

    for i in range(25):
        tx, ty = shift_value_5x5(i, shift_val)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        lf[:, :, i] = cv2.warpAffine(img, M, (256, 256), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return lf

def lf_to_blob(img):
    h, w = img.shape[:2]
    blob = np.zeros((5*5, 3, h//5, w//5))

    for ax in range(5):
        for ay in range(5):
            blob[5*ax+ay, :, :, :] = np.moveaxis(img[ay::5, ax::5, :], -1, 0)

    return blob

def blob_to_lf(blob):
    tensor = blob_to_tensor(blob)
    b, h, w, c = tensor.shape
    img = np.zeros((h*5, w*5, c))

    for ax in range(5):
        for ay in range(5):
            img[ay::5, ax::5, :] = tensor[5*ax+ay, :, :, :]
    if c == 1:
        img = np.squeeze(img, -1)

    return img

def lf_load_5x5(path='./data_input/sai/sai0_{}.png', is_color=True):
    if is_color == True:
        color_mode = cv2.IMREAD_COLOR
        lf_tensor = np.zeros((25, 256, 256, 3))
    elif is_color == False:
        color_mode = cv2.IMREAD_GRAYSCALE
        lf_tensor = np.zeros((25, 256, 256, 1))
    else:
        print('!!! Wrong is_color param !!!')
        exit()

    for i in range(25):
        try:
            img = cv2.imread(path.format(index_picker_5x5(i)), color_mode)
            lf_tensor[i, :, :, :] = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        except ValueError as e:
            img = cv2.imread(path.format(index_picker_5x5(i)), color_mode)          
            lf_tensor[i, :, :, 0] = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
            
    return tensor_to_blob(lf_tensor)

def lf_save_5x5(lf_blob, path='./data_output/sai/sai0_{}.png'):
    lf_tensor = blob_to_tensor(lf_blob)

    for i in range(lf_tensor.shape[0]):
        try:
            cv2.imwrite(path.format(i), lf_tensor[i, :, :, :])
        except ValueError as e:
            cv2.imwrite(path.format(i), lf_tensor[i, :, :, 0])

def blob_to_image(blob):
    tensor = blob_to_tensor(blob)
    b, h, w, c = tensor.shape
    if b != 1 and (c != 3 or c != 1):
        print('!!! Wrong shape !!!')
        exit()
    
    if c == 3:
        img = np.squeeze(tensor, 0)
    else:
        img = np.squeeze(np.squeeze(tensor, 0), -1)

    return img

def image_to_blob(img):
    if len(img.shape) == 3:
        c = 3
    elif len(img.shape) == 2:
        c = 1
    else:
        print('!!! Wrong shape !!!')
        exit()

    if c == 3:
        tensor = np.expand_dims(img, 0)
    else:
        tensor = np.expand_dims(np.expand_dims(img, 0), -1)
    
    return tensor_to_blob(tensor)

def tensor_to_blob(tensor):
    return np.moveaxis(tensor, -1, 1)

def blob_to_tensor(blob):
    return np.moveaxis(blob, 1, -1)

def predict_to_blob(predict):
    b, c, h, w = predict.shape

    if c == 75:
        blob = np.zeros((c//3, 3, h, w))
        for ax in range(5):
            for ay in range(5):
                blob[5*ax+ay, :, :, :] = predict[:, 3*(5*ax+ay):3*(5*ax+ay)+3, :, :]
    elif c == 25:
        blob = np.zeros((c, 1, h, w))
        for i in range(25):
            blob[i, 0, :, :] = predict[:, i, :, :]
    else:
        print('!!! Wrong predict size !!!')
        exit()
        
    return blob

def lf_flip_hor(blob):
    dst_blob = np.zeros((blob.shape))
    for ax in range(5):
        for ay in range(5):
            dst_blob[5*ax+ay, :, :, :] = blob[5*(4-ax)+ay, :, :, :]
    return dst_blob

def lf_flip_ver(blob):
    dst_blob = np.zeros((blob.shape))
    for ax in range(5):
        for ay in range(5):
            dst_blob[5*ax+ay, :, :, :] = blob[5*ax+(4-ay), :, :, :]
    return dst_blob

def lf_transpose(blob):
    dst_blob = np.zeros((blob.shape))
    for ax in range(5):
        for ay in range(5):
            dst_blob[5*ax+ay, :, :, :] = blob[5*ay+ax, :, :, :]
    return dst_blob           

def epi_slicing(blob, t=2, s=2, v=256//2, u=256//2):
    '''
    v = y coord of image
    u = x coord of image
    t = y coord of SAI gird
    s = x coord of SAI gird
    '''
    
    tensor = blob_to_tensor(blob)
    b, h, w, c = tensor.shape

    # Stack SAI
    stack_ver = np.zeros((5, h, w, c))
    stack_hor = np.zeros((5, h, w, c))
    for i in range(5):
        stack_ver[i, :, :, :] = tensor[i+(5*s), :, :, :]
        stack_hor[i, :, :, :] = tensor[5*i+t, :, :, :]

    # Extract EPI
    epi_ver = np.zeros((h, 5, c))
    epi_hor = np.zeros((5, w, c))
    for i in range(5):
        epi_ver[:, i, :] = stack_ver[i, :, u, :]
        epi_hor[i, :, :] = stack_hor[i, v, :, :]

    #cv2.imshow('test', np.uint8(epi_ver))
    #cv2.waitKey()

    return image_to_blob(epi_hor), image_to_blob(epi_ver)

def view_center_change_5x5(blob, shift_val):
    tensor = blob_to_tensor(blob)
    for i in range(25):
        tx, ty = shift_value_5x5(i, shift_val)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        try:
            tensor[i, :, :, :] = cv2.warpAffine(tensor[i, :, :, :], M, (256, 256), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except ValueError as e:
            tensor[i, :, :, 0] = cv2.warpAffine(tensor[i, :, :, 0], M, (256, 256), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return tensor_to_blob(tensor)

def psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100.
    return 10 * math.log10(255. * 255. / mse)

def blob_to_grid(blob):
    tensor = blob_to_tensor(blob)
    c = tensor.shape[-1]
    grid_img = np.zeros((256*5, 256*5, c))

    for ax in range(5):
        for ay in range(5):
            grid_img[256*ay:256*ay+256, 256*ax:256*ax+256, :] = tensor[5*ax+ay, :, :, :]
            
    if c == 1:
        grid_img = np.squeeze(grid_img, -1)

    return grid_img

def data_preparation():
    n_tot = 600
    n_sai = 25
    directory = './datas/face_dataset/face_train_5x5_f50_b2.5'
    ext_src = '.png'
    ext_dst = '.jpg'

    for i_tot in range(n_tot):
        # Make seperated directory
        directory_new = os.path.join(directory, 'img{}'.format(i_tot))
        if not os.path.exists(directory_new):
            os.makedirs(directory_new)

        # Read and save image at new directory
        for i_sai in range(n_sai):
            i_pick = index_picker_5x5(i_sai, pick_mode='9x9')
            filename_src = 'sai{}_{}'.format(i_tot, i_pick)
            filename_dst = 'sai{}'.format(i_sai)
            path_src = os.path.join(directory, filename_src + ext_src)
            path_dst = os.path.join(directory_new, filename_dst + ext_dst)
            print(path_src, path_dst)
            img = cv2.imread(path_src, cv2.IMREAD_COLOR)
            cv2.imwrite(path_dst, img)
            os.remove(path_src)

def name_erase():
    n_tot = 600
    n_sai = 25
    directory = './datas/face_dataset/face_train_5x5_f50_b2.5'
    
    for i_tot in range(n_tot):
        for i_sai in range(n_sai):
            name_src = os.path.join(directory, 'img{}'.format(i_tot), 'sai{}_{}.jpg'.format(i_tot, i_sai))
            name_dst = os.path.join(directory, 'img{}'.format(i_tot), 'sai{}.jpg'.format(i_sai))
            print(name_src, name_dst)
            os.rename(name_src, name_dst)