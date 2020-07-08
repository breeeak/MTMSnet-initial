import os
import cv2
import glob
import h5py

import random
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.losses import mean_squared_error,binary_crossentropy
import matplotlib.pyplot as plt
from keras.preprocessing import image


def eval_path_files(root_path='H:/label/fabric_for_label', validation_split=0.1):
    """
    把数据集路径生成txt，便于读取。
    数据集放在train_data 和 train_data 下
    输出在root_path/paths_test_train_val
    :param root_path:
    :param validation_split:
    :return:
    """
    paths_train = os.path.join(root_path, 'train_data')
    paths_test = os.path.join(root_path, 'test_data')

    img_paths_train = []
    for img_path in glob.glob(os.path.join(paths_train, '*.bmp')):
        if img_path.endswith(".jpg"):
            h5_path = img_path.replace('.jpg', 'FPALL.h5')
        elif img_path.endswith(".png"):
            h5_path = img_path.replace('.png', 'FPALL.h5')
        elif img_path.endswith(".bmp"):
            h5_path = img_path.replace('.bmp', 'FPALL.h5')
        consist = glob.glob(h5_path)
        if len(consist) <= 0:
            continue
        img_paths_train.append(str(img_path))
    for img_path in glob.glob(os.path.join(paths_train, '*.png')):
        if img_path.endswith(".jpg"):
            h5_path = img_path.replace('.jpg', 'FPALL.h5')
        elif img_path.endswith(".png"):
            h5_path = img_path.replace('.png', 'FPALL.h5')
        elif img_path.endswith(".bmp"):
            h5_path = img_path.replace('.bmp', 'FPALL.h5')

        consist = glob.glob(h5_path)
        if len(consist) <= 0:
            continue
        img_paths_train.append(str(img_path))
    for img_path in glob.glob(os.path.join(paths_train, '*.jpg')):
        if img_path.endswith(".jpg"):
            h5_path = img_path.replace('.jpg', 'FPALL.h5')
        elif img_path.endswith(".png"):
            h5_path = img_path.replace('.png', 'FPALL.h5')
        elif img_path.endswith(".bmp"):
            h5_path = img_path.replace('.bmp', 'FPALL.h5')

        consist = glob.glob(h5_path)
        if len(consist) <= 0:
            continue
        img_paths_train.append(str(img_path))
    print("len(img_paths_train) =", len(img_paths_train))
    img_paths_test = []
    for img_path in glob.glob(os.path.join(paths_test, '*.bmp')):
        if img_path.endswith(".jpg"):
            h5_path = img_path.replace('.jpg', 'FPALL.h5')
        elif img_path.endswith(".png"):
            h5_path = img_path.replace('.png', 'FPALL.h5')
        elif img_path.endswith(".bmp"):
            h5_path = img_path.replace('.bmp', 'FPALL.h5')

        consist = glob.glob(h5_path)
        if len(consist) <= 0:
            continue
        img_paths_test.append(str(img_path))
    for img_path in glob.glob(os.path.join(paths_test, '*.png')):
        if img_path.endswith(".jpg"):
            h5_path = img_path.replace('.jpg', 'FPALL.h5')
        elif img_path.endswith(".png"):
            h5_path = img_path.replace('.png', 'FPALL.h5')
        elif img_path.endswith(".bmp"):
            h5_path = img_path.replace('.bmp', 'FPALL.h5')

        consist = glob.glob(h5_path)
        if len(consist) <= 0:
            continue
        img_paths_test.append(str(img_path))
    for img_path in glob.glob(os.path.join(paths_test, '*.jpg')):
        if img_path.endswith(".jpg"):
            h5_path = img_path.replace('.jpg', 'FPALL.h5')
        elif img_path.endswith(".png"):
            h5_path = img_path.replace('.png', 'FPALL.h5')
        elif img_path.endswith(".bmp"):
            h5_path = img_path.replace('.bmp', 'FPALL.h5')

        consist = glob.glob(h5_path)
        if len(consist) <= 0:
            continue
        img_paths_test.append(str(img_path))
    print("len(img_paths_test) =", len(img_paths_test))

    random.seed(1)
    random.shuffle(img_paths_train)
    lst_to_write = [img_paths_train, img_paths_train[:int(len(img_paths_train)*validation_split)], img_paths_test]
    for idx, i in enumerate(['train', 'val', 'test']):
        with open(root_path+'/paths_test_train_val/paths_'+i+'.txt', 'w') as fout:
            fout.write(str(lst_to_write[idx]))
            print('Writing to /paths_test_train_val/paths_'+i+'.txt')


def gen_paths(root_path='H:/label/fabric_for_label'):
    """
    读取路径txt文件
    :param root_path:
    :return:
    """

    root_path = os.path.join(root_path, "paths_test_train_val")
    img_paths = []
    for i in sorted([os.path.join(root_path, p) for p in os.listdir(root_path)]):
        with open(i, 'r') as fin:
            img_paths.append(eval(fin.read()))
    return img_paths    # img_paths_test, img_paths_train, img_paths_val


def load_img(path):
    """
    加载图片并进行归一化
    :param path:
    :return:
    """
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img[:, :, 0]=(img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1]=(img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2]=(img[:, :, 2] - 0.406) / 0.225
    return img.astype(np.float32)


def inverse_norm_img(pred):
    """
    加载图片并进行归一化
    :param path:
    :return:
    """
    MinValue = np.min(pred[:])
    MaxValue = np.max(pred[:])
    for i in range(len(pred)):
        for j in range(len(pred[0])):
            pred[i,j] = (pred[i,j] - MinValue) / (MaxValue - MinValue)
            pred[i,j] = pred[i,j]*255
    return pred.astype(np.uint8)



def img_from_h5(img_path, labeltypes="warp"):
    """
    读取h5 的label
    :param img_path:
    :return:
    """
    gt_file = h5py.File(img_path, 'r')
    if labeltypes=="warp":
        density_map = np.asarray(gt_file['dencity_map_warp'])
    elif labeltypes=="weft":
        density_map = np.asarray(gt_file['dencity_map_weft'])
    elif labeltypes=="point_warp":
        density_map = np.asarray(gt_file['pattern_warp_map'])
    elif labeltypes=="point_weft":
        density_map = np.asarray(gt_file['pattern_weft_map'])
    elif labeltypes=="point":
        density_map = np.asarray(gt_file['locs'])
    stride = 1
    if stride > 1:
        density_map_stride = np.zeros((np.asarray(density_map.shape).astype(int)//stride).tolist(), dtype=np.float32)
        for r in range(density_map_stride.shape[0]):
            for c in range(density_map_stride.shape[1]):
                density_map_stride[r, c] = np.sum(density_map[r*stride:(r+1)*stride, c*stride:(c+1)*stride])
    else:
        density_map_stride = density_map
    return density_map_stride


def gen_x_y(img_paths, labeltypes="warp", isTrain=True):
    """
    读取图片并进行归一化，并增广图像
    :param img_paths:
    :return:
    """
    if isTrain:
        x, y = [], []
        idx_shuffle = list(range(len(img_paths)))
        random.shuffle(idx_shuffle)
        img_paths = np.array(img_paths)[idx_shuffle].tolist()
        for i in img_paths:
            if i.endswith(".jpg"):
                h5_path = i.replace('.jpg', 'FPALL.h5')
            elif i.endswith(".png"):
                h5_path = i.replace('.png', 'FPALL.h5')
            elif i.endswith(".bmp"):
                h5_path = i.replace('.bmp', 'FPALL.h5')

            consist = glob.glob(h5_path)
            if len(consist) <= 0:
                continue
            x_ = load_img(i)
            x.append(x_)
            y_ = img_from_h5(h5_path, labeltypes="warp")
            y_2 = img_from_h5(h5_path, labeltypes="weft")
            y_3 = img_from_h5(h5_path, labeltypes="point_warp")
            y_4 = img_from_h5(h5_path, labeltypes="point_weft")
            yy = np.dstack((y_, y_2,y_3, y_4))
            y.append(yy)
        idx_shuffle = list(range(len(x)))
        random.shuffle(idx_shuffle)
        x = [x[idx] for idx in idx_shuffle]
        y = [y[idx] for idx in idx_shuffle]
        return x, y, img_paths
    else:
        x, y = [], []
        idx_shuffle = list(range(len(img_paths)))
        random.shuffle(idx_shuffle)
        img_paths = np.array(img_paths)[idx_shuffle].tolist()
        for i in img_paths:
            if i.endswith(".jpg"):
                h5_path = i.replace('.jpg', 'FPALL.h5')
            elif i.endswith(".png"):
                h5_path = i.replace('.png', 'FPALL.h5')
            elif i.endswith(".bmp"):
                h5_path = i.replace('.bmp', 'FPALL.h5')
            consist = glob.glob(h5_path)
            if len(consist) <= 0:
                continue
            x_ = load_img(i)
            x.append(x_)
            y_ = img_from_h5(h5_path, labeltypes=labeltypes)
            y.append(np.expand_dims(y_, axis=-1))
        return x, y, img_paths


def generator_x_y(img_paths, labeltypes="warp", batch_size=1,isTrain=True):
    datagen = image.ImageDataGenerator(fill_mode='wrap', horizontal_flip=True, vertical_flip=True,channel_shift_range=10)
    while True:
        # 第0.25种处理方式 缩小
        for i in img_paths:
            x, y1, y2 = [], [], []
            if i.endswith(".jpg"):
                h5_path = i.replace('.jpg', 'FPALL.h5')
            elif i.endswith(".png"):
                h5_path = i.replace('.png', 'FPALL.h5')
            elif i.endswith(".bmp"):
                h5_path = i.replace('.bmp', 'FPALL.h5')
            consist = glob.glob(h5_path)
            if len(consist) <= 0:
                continue
            x_ = load_img(i)
            x_ = datagen.apply_transform(x_, {"zx": 0.25})
            x_ = datagen.apply_transform(x_, {"zy": 0.25})
            x.append(x_)
            y_ = img_from_h5(h5_path, labeltypes="warp")
            y_2 = img_from_h5(h5_path, labeltypes="weft")
            y_3 = img_from_h5(h5_path, labeltypes="point_warp")
            y_4 = img_from_h5(h5_path, labeltypes="point_weft")

            y_ = datagen.apply_transform(np.expand_dims(y_, axis=-1), {"zx": 0.25 })
            y_ = np.squeeze(datagen.apply_transform(y_, {"zy": 0.25}))
            y_2 = datagen.apply_transform(np.expand_dims(y_2, axis=-1), {"zx": 0.25 })
            y_2 = np.squeeze(datagen.apply_transform(y_2, {"zy": 0.25}))
            y_3 = datagen.apply_transform(np.expand_dims(y_3, axis=-1), {"zx": 0.25 })
            y_3 = np.squeeze(datagen.apply_transform(y_3, {"zy": 0.25}))
            y_4 = datagen.apply_transform(np.expand_dims(y_4, axis=-1), {"zx": 0.25 })
            y_4 = np.squeeze(datagen.apply_transform(y_4, {"zy": 0.25}))


            yy1 = np.dstack((y_, y_2))
            yy2 = np.dstack((y_3, y_4))
            y1.append(yy1)
            y2.append(yy2)
            y = [y1, y2]
            yield np.array(x), y
        # 第0.5种处理方式 缩小
        for i in img_paths:
            x, y1, y2 = [], [], []
            if i.endswith(".jpg"):
                h5_path = i.replace('.jpg', 'FPALL.h5')
            elif i.endswith(".png"):
                h5_path = i.replace('.png', 'FPALL.h5')
            elif i.endswith(".bmp"):
                h5_path = i.replace('.bmp', 'FPALL.h5')
            consist = glob.glob(h5_path)
            if len(consist) <= 0:
                continue
            x_ = load_img(i)
            x_ = datagen.apply_transform(x_, {"zx": 0.4})
            x_ = datagen.apply_transform(x_, {"zy": 0.4})
            x.append(x_)
            y_ = img_from_h5(h5_path, labeltypes="warp")
            y_2 = img_from_h5(h5_path, labeltypes="weft")
            y_3 = img_from_h5(h5_path, labeltypes="point_warp")
            y_4 = img_from_h5(h5_path, labeltypes="point_weft")

            y_ = datagen.apply_transform(np.expand_dims(y_, axis=-1), {"zx": 0.4})
            y_ = np.squeeze(datagen.apply_transform(y_, {"zy": 0.4}))
            y_2 = datagen.apply_transform(np.expand_dims(y_2, axis=-1), {"zx": 0.4})
            y_2 = np.squeeze(datagen.apply_transform(y_2, {"zy": 0.4}))
            y_3 = datagen.apply_transform(np.expand_dims(y_3, axis=-1), {"zx": 0.4})
            y_3 = np.squeeze(datagen.apply_transform(y_3, {"zy": 0.4}))
            y_4 = datagen.apply_transform(np.expand_dims(y_4, axis=-1), {"zx": 0.4})
            y_4 = np.squeeze(datagen.apply_transform(y_4, {"zy": 0.4}))


            yy1 = np.dstack((y_, y_2))
            yy2 = np.dstack((y_3, y_4))
            y1.append(yy1)
            y2.append(yy2)
            y = [y1, y2]
            yield np.array(x), y
        # 第0.5种处理方式 缩小
        for i in img_paths:
            x, y1, y2 = [], [], []
            if i.endswith(".jpg"):
                h5_path = i.replace('.jpg', 'FPALL.h5')
            elif i.endswith(".png"):
                h5_path = i.replace('.png', 'FPALL.h5')
            elif i.endswith(".bmp"):
                h5_path = i.replace('.bmp', 'FPALL.h5')
            consist = glob.glob(h5_path)
            if len(consist) <= 0:
                continue
            x_ = load_img(i)
            x_ = datagen.apply_transform(x_, {"zx": 0.3})
            x_ = datagen.apply_transform(x_, {"zy": 0.3})
            x.append(x_)
            y_ = img_from_h5(h5_path, labeltypes="warp")
            y_2 = img_from_h5(h5_path, labeltypes="weft")
            y_3 = img_from_h5(h5_path, labeltypes="point_warp")
            y_4 = img_from_h5(h5_path, labeltypes="point_weft")

            y_ = datagen.apply_transform(np.expand_dims(y_, axis=-1), {"zx": 0.3})
            y_ = np.squeeze(datagen.apply_transform(y_, {"zy": 0.3}))
            y_2 = datagen.apply_transform(np.expand_dims(y_2, axis=-1), {"zx": 0.3})
            y_2 = np.squeeze(datagen.apply_transform(y_2, {"zy": 0.3}))
            y_3 = datagen.apply_transform(np.expand_dims(y_3, axis=-1), {"zx": 0.3})
            y_3 = np.squeeze(datagen.apply_transform(y_3, {"zy": 0.3}))
            y_4 = datagen.apply_transform(np.expand_dims(y_4, axis=-1), {"zx": 0.3})
            y_4 = np.squeeze(datagen.apply_transform(y_4, {"zy": 0.3}))


            yy1 = np.dstack((y_, y_2))
            yy2 = np.dstack((y_3, y_4))
            y1.append(yy1)
            y2.append(yy2)
            y = [y1, y2]
            yield np.array(x), y





def plot_pridict(epoch, logs, model,path_val_display,x_val_display, y_val_display):
    pred = model.predict(np.expand_dims(x_val_display, axis=0))
    pred_density, pred_pattern = np.squeeze(pred[0]), np.squeeze(pred[1])
    pred_density_warp, pred_density_weft = np.dsplit(pred_density, 2)
    pred_pattern_warp, pred_pattern_weft = np.dsplit(pred_pattern, 2)
    # real_warp, real_weft, real_warp_point, real_weft_point = np.dsplit(y_val_display, 4)
    
    fg, (ax_x_ori, ax_y, ax_pred) = plt.subplots(1, 3, figsize=(20, 4))
    # 原图
    ax_x_ori.imshow(cv2.cvtColor(cv2.imread(path_val_display), cv2.COLOR_BGR2RGB))
    ax_x_ori.set_title('Original Image' + str(path_val_display))
    # 真值
    ax_y.imshow(np.squeeze(pred_density_warp), cmap=plt.cm.jet)
    ax_y.set_title('pred_density_warp')
    # 预测值
    ax_pred.imshow(np.squeeze(pred_density_weft), cmap=plt.cm.jet)
    ax_pred.set_title('pred_density_weft')
    plt.suptitle('epoch/loss = ' + str(epoch) + "/" + str(logs["loss"]))
    plt.show()

    fg, (ax_x_ori, ax_y, ax_pred) = plt.subplots(1, 3, figsize=(20, 4))
    # 原图
    ax_x_ori.imshow(cv2.cvtColor(cv2.imread(path_val_display), cv2.COLOR_BGR2RGB))
    ax_x_ori.set_title('Original Image2' + str(path_val_display))
    # 真值
    ax_y.imshow(np.squeeze(pred_pattern_warp), cmap=plt.cm.jet)
    ax_y.set_title('pred_pattern_warp')
    # 预测值
    ax_pred.imshow(np.squeeze(pred_pattern_weft), cmap=plt.cm.jet)
    ax_pred.set_title('pred_pattern_weft')
    plt.suptitle('epoch/loss = ' + str(epoch) + "/" + str(logs["loss"]))
    plt.show()

def ssim_loss(y_true, y_pred, c1=0.01**2, c2=0.03**2):
    """
    相当于loss 的正则项
    :param y_true:
    :param y_pred:
    :param c1:
    :param c2:
    :return:
    """
    # Generate a 11x11 Gaussian kernel with standard deviation of 1.5
    y_pred, y_pred1 = tf.split(y_pred, 2, 3)
    y_true, y_true1= tf.split(y_true, 2, 3)
    weights_initial = np.multiply(
        cv2.getGaussianKernel(11, 1.5),
        cv2.getGaussianKernel(11, 1.5).T
    )

    # weights_initial = np.stack((weights_initial,weights_initial),axis=2)

    weights_initial = weights_initial.reshape(*weights_initial.shape,1, 1)
    weights_initial = K.cast(weights_initial, tf.float32)

    # warp
    mu_F = tf.nn.conv2d(y_pred, weights_initial, [1, 1, 1, 1], padding='SAME')
    mu_Y = tf.nn.conv2d(y_true, weights_initial, [1, 1, 1, 1], padding='SAME')
    mu_F_mu_Y = tf.multiply(mu_F, mu_Y)
    mu_F_squared = tf.multiply(mu_F, mu_F)
    mu_Y_squared = tf.multiply(mu_Y, mu_Y)
    sigma_F_squared = tf.nn.conv2d(tf.multiply(y_pred, y_pred), weights_initial, [1, 1, 1, 1], padding='SAME') - mu_F_squared
    sigma_Y_squared = tf.nn.conv2d(tf.multiply(y_true, y_true), weights_initial, [1, 1, 1, 1], padding='SAME') - mu_Y_squared
    sigma_F_Y = tf.nn.conv2d(tf.multiply(y_pred, y_true), weights_initial, [1, 1, 1, 1], padding='SAME') - mu_F_mu_Y
    ssim = ((2 * mu_F_mu_Y + c1) * (2 * sigma_F_Y + c2)) / ((mu_F_squared + mu_Y_squared + c1) * (sigma_F_squared + sigma_Y_squared + c2))

    # weft
    mu_F = tf.nn.conv2d(y_pred1, weights_initial, [1, 1, 1, 1], padding='SAME')
    mu_Y = tf.nn.conv2d(y_true1, weights_initial, [1, 1, 1, 1], padding='SAME')
    mu_F_mu_Y = tf.multiply(mu_F, mu_Y)
    mu_F_squared = tf.multiply(mu_F, mu_F)
    mu_Y_squared = tf.multiply(mu_Y, mu_Y)

    sigma_F_squared = tf.nn.conv2d(tf.multiply(y_pred1, y_pred1), weights_initial, [1, 1, 1, 1],
                                   padding='SAME') - mu_F_squared
    sigma_Y_squared = tf.nn.conv2d(tf.multiply(y_true1, y_true1), weights_initial, [1, 1, 1, 1],
                                   padding='SAME') - mu_Y_squared
    sigma_F_Y = tf.nn.conv2d(tf.multiply(y_pred1, y_true1), weights_initial, [1, 1, 1, 1], padding='SAME') - mu_F_mu_Y

    ssim1 = ((2 * mu_F_mu_Y + c1) * (2 * sigma_F_Y + c2)) / (
                (mu_F_squared + mu_Y_squared + c1) * (sigma_F_squared + sigma_Y_squared + c2))


    all_sim = 1 - tf.reduce_mean(ssim, reduction_indices=[1, 2, 3]) + 1 - tf.reduce_mean(ssim1, reduction_indices=[1, 2, 3])



    return all_sim


def ssim_eucli_loss(y_true, y_pred, alpha=0.001):
    """
    损失函数
    :param y_true:
    :param y_pred:
    :param alpha:
    :return:
    """
    ssim = ssim_loss(y_true, y_pred)
    eucli = mean_squared_error(y_true, y_pred)
    # eucli = binary_crossentropy(y_true, y_pred)
    loss = eucli + alpha * ssim
    return loss

def eucli_loss(y_true, y_pred):
    """
    损失函数
    :param y_true:
    :param y_pred:
    :param alpha:
    :return:
    """
    eucli = binary_crossentropy(y_true, y_pred)
    return eucli


def eval_loss(model, x, y):
    """
    计算损失值
    :param model:
    :param x:
    :param y:
    :return:
    """
    preds = []
    for i in x:
        preds.append(np.squeeze(model.predict(i)))
    labels = []
    for i in y:
        labels.append(np.squeeze(i))
    losses_DMD = []
    for i in range(len(preds)):
        losses_DMD.append(np.sum(np.abs(preds[i] - labels[i])))
    loss_DMD = np.mean(losses_DMD)
    losses_MAE = []
    for i in range(len(preds)):
        losses_MAE.append(np.abs(np.sum(preds[i]) - np.sum(labels[i])))
    loss_DMD = np.mean(losses_DMD)
    loss_MAE = np.mean(losses_MAE)
    return loss_DMD, loss_MAE


def focal_loss(gamma=2., alpha=.5):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def calc_dist(pred, yarn_type="weft", n=19, line_num=10):
    yarn_loc = []
    yarn_val = []
    w = len(pred[0])
    h = len(pred)
    if yarn_type == "weft":
        line_gap = int(w/(line_num+1))
        jloc = 0
        for j in range(line_num):
            jloc = jloc + line_gap
            for i in range(h):
                if i-n >= 0 and i+n < h:
                    if pred[i, jloc] == np.max(pred[i-n:i+n, jloc]):
                        yarn_loc.append([i, jloc])
                        yarn_val.append(pred[i, jloc])
                elif i-n <0:
                    if pred[i, jloc] == np.max(pred[0:i+n, jloc]):
                        yarn_loc.append([i, jloc])
                        yarn_val.append(pred[i, jloc])
                elif i+n >= h:
                    if pred[i, jloc] == np.max(pred[i-n:h-1, jloc]):
                        yarn_loc.append([i, jloc])
                        yarn_val.append(pred[i, jloc])
        yarn_loc = np.array(yarn_loc)
        yarn_val = np.array(yarn_val)
        return np.array(yarn_loc), np.array(yarn_val)


def img_kmean(array, main_color_num):
    """
    Use Kmean to float32 arry and return Main Color Percenage, rgb

    Param:
        array: np.float32 array
        main_color_num: Main Color Number
    Return:
        main_color_per_rgb: [[percentage1, [r1, g1, b1]],[percentage2, [r2, g2, b2]],...]
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(array, main_color_num,
                                    None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    main_color_per_rgb, i = [], 0
    while i < main_color_num:
        current_color_percent = len(array[label.ravel() == i]) / len(label)
        main_color_per_rgb.append([
            current_color_percent,
            [int(center[i][0]),
                int(center[i][1]),
                int(center[i][2])]
        ])
        i += 1
    main_color_per_rgb.sort(reverse=True, key=lambda n: n[0])
    return main_color_per_rgb


if __name__ == '__main__':
    eval_path_files('I:\\label\\imgforpoint\\test\\tobeused\\zoomall')