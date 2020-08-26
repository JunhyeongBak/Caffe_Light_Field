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

def sais_to_lf():
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

def lf_to_sais():
    TOT = 77
    ANGULAR_RES = 14
    TARGET_RES = 5
    LF_WIDTH = 7574  #3584
    LF_HEIGHT = 5264 #2688
    SAI_HEIGHT = 376 # 192 # int(LF_HEIGHT / ANGULAR_RES) # SPATIAL_HEIGHT
    SAI_WIDTH = 541 # 256 # int(LF_WIDTH / ANGULAR_RES) # SPATIAL_WIDTH
    CH = 3

    for i_tot in tqdm(range(TOT)):
        lf = cv2.imread('./utils/lf/lf'+str(i_tot)+'.png', cv2.IMREAD_COLOR)
        full_sais = np.zeros((SAI_HEIGHT, SAI_WIDTH, CH, ANGULAR_RES, ANGULAR_RES))
        for ax in range(ANGULAR_RES):
            for ay in range(ANGULAR_RES):
                sai_resize = cv2.resize(lf[ay::ANGULAR_RES, ax::ANGULAR_RES, :], dsize=(SAI_WIDTH, SAI_HEIGHT), interpolation=cv2.INTER_CUBIC)
                full_sais[:, :, :, ay, ax] = sai_resize

        crop_sais = full_sais[:, :, :, ((ANGULAR_RES-TARGET_RES)//2):((ANGULAR_RES-TARGET_RES)//2)+TARGET_RES, ((ANGULAR_RES-TARGET_RES)//2):((ANGULAR_RES-TARGET_RES)//2)+TARGET_RES]
        sais_list = np.zeros((SAI_HEIGHT, SAI_WIDTH, CH, TARGET_RES*TARGET_RES))
        sai_cnt = 0
        for ax in range(TARGET_RES):
            for ay in range(TARGET_RES):
                #sais_list[:, :, :, sai_cnt] = crop_sais[:, :, :, ay, ax]
                cv2.imwrite('./utils/sai_val/sai'+str(i_tot)+'_'+str(sai_cnt)+'.png', crop_sais[:, ((SAI_WIDTH-SAI_HEIGHT)//2):((SAI_WIDTH-SAI_HEIGHT)//2)+SAI_HEIGHT, :, ay, ax])
                sai_cnt = sai_cnt + 1

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

def renamer_glob():
    i_cnt = 0
    for path in glob.glob('./utils/lf_raw/org/*.png'):
        lf = cv2.imread(path, cv2.IMREAD_COLOR)
        cv2.imwrite('./utils/lf_raw/lf_raw'+str(i_cnt)+'.png', lf)
        i_cnt = i_cnt + 1

def flower_cropper():
    TOT = 3343
    SAI = 25
    for i_tot in tqdm(range(TOT)):
        crop_point = np.random.randint(10, 54+1)
        for i_sai in range(SAI):
            img_src = cv2.imread('./datasets/SAIs/sai'+str(i_tot)+'_'+str(i_sai)+'.png', cv2.IMREAD_COLOR)
            img_dst = img_src[:, crop_point:crop_point+192, :]
            #print('none flip', crop_point)
            cv2.imwrite('./datasets/SAIs_Crop/sai'+str(i_tot+3343+3343)+'_'+str(i_sai)+'.png', img_dst)
            
if __name__ == '__main__':
    #dataset_resize()
    #sais_to_lf()
    #gif_maker()
    #renamer()
    #renamer_glob()
    #lf_to_sais()