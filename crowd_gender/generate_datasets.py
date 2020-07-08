import os
import cv2
import glob
import h5py
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
from crowd_gender.utils import get_density_map_gaussian
import matplotlib.pyplot as plt

def gen_h5_ground(root,is_gen_gender=False):

    part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
    part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
    part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
    part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')
    path_sets_train = [part_A_train, part_B_train]
    path_sets_test = [part_A_test, part_B_test]

    img_paths_train = []
    for path in path_sets_train:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths_train.append(img_path)
    print(len(img_paths_train))
    img_paths_test = []
    for path in path_sets_test:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths_test.append(img_path)
    print(len(img_paths_test))

    for part in ['A', 'B']:
        for t in ['train', 'test']:
            if not os.path.exists(os.path.join(root, 'part_{}_final/{}_data'.format(part, t), 'ground')):
                os.mkdir(os.path.join(root, 'part_{}_final/{}_data'.format(part, t), 'ground'))

    for img_paths in [img_paths_train, img_paths_test]:
        for img_path in tqdm(img_paths):

            pts = loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_',
                                                                                                     'GT_IMG_'))  # 读取mat文件
            img = cv2.imread(img_path)

            k = np.zeros((img.shape[0], img.shape[1]))
            gt = pts["image_info"][0, 0][0, 0][0]
            if is_gen_gender:
                pts_gender = loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth_gender').replace(
                    'IMG_', 'GD_GT_IMG_'))  # 读取mat文件
                gt_gender = pts_gender["gender_info"][0,0][0]
                k1 = np.zeros((img.shape[0], img.shape[1]))
                k2 = np.zeros((img.shape[0], img.shape[1]))

                gt_male = []
                gt_female =[]
            for i in range(len(gt)):
                if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1
                    if is_gen_gender:
                        if gt_gender[i]==1:
                            k1[int(gt[i][1]), int(gt[i][0])] = 1
                            gt_male.append(gt[i])
                        elif gt_gender[i]==2:
                            k2[int(gt[i][1]), int(gt[i][0])] = 1
                            gt_female.append(gt[i])
            k_all = get_density_map_gaussian(k, gt, adaptive_kernel=True)
            if is_gen_gender:
                k_male = get_density_map_gaussian(k1, gt, adaptive_kernel=True)
                k_female = get_density_map_gaussian(k2, gt, adaptive_kernel=True)

            file_path = img_path.replace('.jpg', '.h5').replace('images', 'ground')

            with h5py.File(file_path, 'w') as hf:
                hf['density'] = k_all
                if is_gen_gender:
                    hf['male_density'] = k_male
                    hf['female_density'] = k_female

def loadtest(root):
    val = loadmat(root)
    pass

def gen_part(root,is_gen_gender=False):


    for gender_path in tqdm(glob.glob(os.path.join(root, '*.mat'))):


        pts = loadmat(gender_path.replace('ground_truth_gender', 'ground_truth').replace('GD_GT_IMG_',
                                                                                                     'GT_IMG_'))  # 读取mat文件
        img = cv2.imread(gender_path.replace('.mat', '.jpg').replace('ground_truth_gender', 'images').replace('GD_GT_IMG_',
                                                                                                     'IMG_'))
        k = np.zeros((img.shape[0], img.shape[1]))
        gt = pts["image_info"][0, 0][0, 0][0]
        if is_gen_gender:
            pts_gender = loadmat(gender_path)  # 读取mat文件
            gt_gender = pts_gender["gender_info"][0,0][0]
            k1 = np.zeros((img.shape[0], img.shape[1]))
            k2 = np.zeros((img.shape[0], img.shape[1]))
            gt_male = []
            gt_female =[]

        for i in range(len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
                if is_gen_gender:
                    if gt_gender[i]==1:
                        k1[int(gt[i][1]), int(gt[i][0])] = 1
                        gt_male.append(gt[i])
                    elif gt_gender[i]==2:
                        k2[int(gt[i][1]), int(gt[i][0])] = 1
                        gt_female.append(gt[i])
        k_all = get_density_map_gaussian(k, gt, adaptive_kernel=True)
        if is_gen_gender:
            k_male = get_density_map_gaussian(k1, gt_male, adaptive_kernel=True)
            k_female = get_density_map_gaussian(k2, gt_female, adaptive_kernel=True)

        file_path = gender_path.replace('.mat', '.h5').replace('ground_truth_gender', 'ground')

        with h5py.File(file_path, 'w') as hf:
            hf['density'] = k_all
            if is_gen_gender:
                hf['male_density'] = k_male
                hf['female_density'] = k_female

    pass


if __name__ == '__main__':
    #gen_h5_ground(root = 'O:/dataset/crowd/ShanghaiTech_Crowd_Counting_Dataset')
    gen_part('O:/dataset/crowd/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/train_data/ground_truth_gender',is_gen_gender=True)
    pass