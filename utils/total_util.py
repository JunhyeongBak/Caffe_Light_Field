import numpy as np
import cv2
from tqdm import tqdm
import glob
import imageio
import warnings

def dataset_resize():
    TOT = 4
    SAI = 25
    HEIGHT = 192
    WIDTH = 192
    for i_tot in tqdm(range(TOT)):
        for i_sai in range(SAI):
            src_img = cv2.imread('./utils/sai_big/sai'+str(i_tot)+'_'+str(i_sai)+'.png', cv2.IMREAD_COLOR)
            dst_img = cv2.resize(src_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('./utils/sai_small/sai'+str(i_tot)+'_'+str(i_sai)+'.png', dst_img)

def sai_to_lf():
    TOT = 4
    SAI_HEIGHT = 192
    SAI_WIDTH = 192
    CH = 3
    SIZE_UP = 3
    ANGULAR_RES = 5 # fixed
    LF_HEIGHT = SAI_HEIGHT*ANGULAR_RES*SIZE_UP
    LF_WIDTH = SAI_WIDTH*ANGULAR_RES*SIZE_UP
    
    for i_tot in tqdm(range(TOT)):
        imgSAIs = np.zeros((SAI_HEIGHT*SIZE_UP, SAI_WIDTH*SIZE_UP, CH, ANGULAR_RES*ANGULAR_RES))
        for i in range(ANGULAR_RES*ANGULAR_RES):
            imgSAI = cv2.imread('./utils/sai_small/sai'+str(i_tot)+'_'+str(i)+'.png', cv2.IMREAD_COLOR)
            imgSAI = cv2.resize(imgSAI, dsize=(SAI_WIDTH*SIZE_UP, SAI_HEIGHT*SIZE_UP), interpolation=cv2.INTER_CUBIC) 
            imgSAIs[:, :, :, i] = imgSAI

        # trans order
        imgSAIs2 = np.zeros((imgSAIs.shape))
        for i in range(ANGULAR_RES*ANGULAR_RES):
            if i == 0 or (i)%ANGULAR_RES==0:
                imgSAIs2[:, :, :, i//ANGULAR_RES] = imgSAIs[:, :, :, i]
            elif i == 1 or (i-1)%ANGULAR_RES==0:
                imgSAIs2[:, :, :, (i-1)//ANGULAR_RES+ANGULAR_RES] = imgSAIs[:, :, :, i]
            elif i == 2 or (i-2)%ANGULAR_RES==0:
                imgSAIs2[:, :, :, (i-2)//ANGULAR_RES+ANGULAR_RES*2] = imgSAIs[:, :, :, i]
            elif i == 3 or (i-3)%ANGULAR_RES==0:
                imgSAIs2[:, :, :, (i-3)//ANGULAR_RES+ANGULAR_RES*3] = imgSAIs[:, :, :, i]
            elif i == 4 or (i-4)%5==0:
                imgSAIs2[:, :, :, (i-4)//ANGULAR_RES+ANGULAR_RES*4] = imgSAIs[:, :, :, i]

        # trans SAI to LF
        imgLF = np.zeros((LF_HEIGHT, LF_WIDTH, CH))
        full_LF_crop = np.zeros((SAI_HEIGHT*SIZE_UP, SAI_WIDTH*SIZE_UP, CH, ANGULAR_RES, ANGULAR_RES))
        
        for ax in range(ANGULAR_RES):
            for ay in range(ANGULAR_RES):
                full_LF_crop[:, :, :, ax, ay] = imgSAIs2[:, :, :, ANGULAR_RES*ax+ay]
        
        for ax in range(ANGULAR_RES):
            for ay in range(ANGULAR_RES):
                imgLF[ay::ANGULAR_RES, ax::ANGULAR_RES, :] = full_LF_crop[:, :, :, ay, ax]

        cv2.imwrite('./utils/lf/lf'+str(i_tot)+'.jpg', imgLF)

def gif_maker():
    TOT = 1
    SAI = 25
    for i_tot in tqdm(range(TOT)):
        img_list = []
        for i_sai in range(SAI):
            img_src = cv2.imread('./utils/sai_est/sai'+str(i_tot)+'_'+str(i_sai)+'.png', cv2.IMREAD_COLOR)
            img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
            img_list.append(img_src)
        imageio.mimsave('./utils/gif/gif'+str(i_tot)+'.gif', img_list, duration=0.2)

def renamer():
    TOT = 1
    SAI = 25
    for i_tot in tqdm(range(TOT)):
        for i_sai in range(SAI):
            img_src = cv2.imread('./utils/sai_est/sai'+str(i_sai)+'.png', cv2.IMREAD_COLOR)
            cv2.imwrite('./utils/sai_est/sai'+str(i_tot)+'_'+str(i_sai)+'.png', img_src)

if __name__ == '__main__':
    #dataset_resize()
    #sai_to_lf()
    gif_maker()
    #renamer()