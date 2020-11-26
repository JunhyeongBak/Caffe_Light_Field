import numpy as np
import cv2
import glob
from tqdm import tqdm

def color_to_gray():
    img_list = glob.glob('./datas/face_dataset/face_train_9x9/*.png')

    for img_path in tqdm(img_list):
        img_src = cv2.imread(img_path, 0)
        cv2.imwrite(img_path, img_src)

def calc_mean():
    img_list = glob.glob('./datas/face_dataset/face_train_9x9/*.png')
    tot_mean = 0.

    for img_path in tqdm(img_list):
        img_src = cv2.imread(img_path, 0)
        img_mean = np.mean(img_src)/255.
        tot_mean = tot_mean+img_mean
    
    print('result: ' + (tot_mean/len(img_list)*255.))
    # 124.3356499444686


if __name__ == "__main__":
    calc_mean()