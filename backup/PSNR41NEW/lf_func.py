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