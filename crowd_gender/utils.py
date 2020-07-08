import os
import cv2
import glob
import h5py
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.losses import mean_squared_error
from keras.preprocessing import image


def load_img(path):
    """
    根据路径读取图片，像素归一化并对图片进行白化，
    即让像素点的平均值为0，方差为1。
    这样做是为了减小图片的范围，使得图片的特征更易于学习
    :param path:
    :return:
    """
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img[:, :, 0]=(img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1]=(img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2]=(img[:, :, 2] - 0.406) / 0.225
    return img.astype(np.float32)


def img_from_h5(img_path, labeltypes="density"):
    """
    读取h5 的label
    :param img_path:
    :param labeltypes: density density_male density_female
    :return:
    """
    gt_file = h5py.File(img_path, 'r')
    density_map = np.asarray(gt_file[labeltypes])
    stride = 1
    if stride > 1:
        density_map_stride = np.zeros((np.asarray(density_map.shape).astype(int)//stride).tolist(), dtype=np.float32)
        for r in range(density_map_stride.shape[0]):
            for c in range(density_map_stride.shape[1]):
                density_map_stride[r, c] = np.sum(density_map[r*stride:(r+1)*stride, c*stride:(c+1)*stride])
    else:
        density_map_stride = density_map
    return density_map_stride


def plot_pridict(epoch, logs, model, path_val_display, x_val_display, y_val_display):
    pred = model.predict(np.expand_dims(x_val_display, axis=0))

    fg, (ax_x_ori, ax_y, ax_pred) = plt.subplots(1, 3, figsize=(20, 4))
    # 原图
    ax_x_ori.imshow(cv2.cvtColor(cv2.imread(path_val_display), cv2.COLOR_BGR2RGB))
    ax_x_ori.set_title('Original Image')
    # 真值
    ax_y.imshow(np.squeeze(pred))
    ax_y.set_title('Prediction: ' + str(np.sum(pred)))
    # 预测值
    ax_pred.imshow(np.squeeze(y_val_display))
    ax_pred.set_title('Ground_truth: ' + str(np.sum(y_val_display)))
    plt.suptitle('epoch/loss = ' + str(epoch) + "/" + str(logs["loss"]))
    plt.show()


def generator_x_y(img_paths,labeltypes="density",isTrain=True):
    datagen = image.ImageDataGenerator(fill_mode='wrap', horizontal_flip=True, vertical_flip=True,channel_shift_range=10)
    while True:

        for i in img_paths:
            x, y = [], []
            if isTrain:
                h5_path = str(i).replace('.jpg', '.h5').replace('images',
                                                                            'train_data\\maps_fixed_kernel')
            else:
                h5_path = str(i).replace('.jpg', '.h5').replace('images',
                                                                            'test_data\\maps_fixed_kernel')
            consist = glob.glob(h5_path)
            if len(consist) <= 0:
                continue
            x_ = load_img(i)
            x.append(np.expand_dims(x_, axis=0))
            y_ = img_from_h5(h5_path, labeltypes=labeltypes)
            y.append(np.expand_dims(y_, axis=-1))
            y.append(np.expand_dims(np.expand_dims(y_, axis=0), axis=-1))
            yield np.array(x), np.array(y)
        # 垂直翻转
        for i in img_paths:
            x, y = [], []
            if isTrain:
                h5_path = str(i).replace('.jpg', '.h5').replace('images',
                                                                            'train_data\\maps_fixed_kernel')
            else:
                h5_path = str(i).replace('.jpg', '.h5').replace('images',
                                                                            'test_data\\maps_fixed_kernel')
            consist = glob.glob(h5_path)
            if len(consist) <= 0:
                continue
            x_ = load_img(i)
            x_ = datagen.apply_transform(x_, {"flip_vertical": True})
            x.append(x_)
            y_ = img_from_h5(h5_path, labeltypes=labeltypes)
            y_ = datagen.apply_transform(y_, {"flip_vertical": True})
            y.append(np.expand_dims(y_, axis=-1))

            yield np.array(x), np.array(y)

        # 水平翻转
        for i in img_paths:
            x, y = [], []
            if isTrain:
                h5_path = str(i).replace('.jpg', '.h5').replace('images',
                                                                'train_data\\maps_fixed_kernel')
            else:
                h5_path = str(i).replace('.jpg', '.h5').replace('images',
                                                                'test_data\\maps_fixed_kernel')
            consist = glob.glob(h5_path)
            if len(consist) <= 0:
                continue
            x_ = load_img(i)
            x_ = datagen.apply_transform(x_, {"flip_horizontal": True})
            x.append(x_)
            y_ = img_from_h5(h5_path, labeltypes=labeltypes)
            y_ = datagen.apply_transform(y_, {"flip_horizontal": True})
            y.append(np.expand_dims(y_, axis=-1))

            yield np.array(x), np.array(y)



def eval_loss(model, x, y):
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



def gen_paths(root_path='H:/label/fabric_for_label'):
    """
    读取路径txt文件
    :param root_path:
    :return:
    """

    root_path = os.path.join(root_path, "paths_train_val_test")
    img_paths = []
    for i in sorted([os.path.join(root_path, p) for p in os.listdir(root_path)]):
        with open(i, 'r') as fin:
            img_paths.append(eval(fin.read()))
    return img_paths    # img_paths_test, img_paths_train, img_paths_val


def eval_path_files(root = 'data/ShanghaiTech/', validation_split=0.05):
    """
    生成训练集和测试集的路径 保存到txt中 便于读取
    :param dataset:
    :param validation_split:
    :return:
    """

    paths_train = os.path.join(root,  'train_data', 'maps_fixed_kernel')
    paths_test = os.path.join(root, 'test_data', 'maps_fixed_kernel')

    img_paths_train = []
    for img_path in glob.glob(os.path.join(paths_train, '*.h5')):
        img_paths_train.append(str(img_path).replace('.h5','.jpg').replace('train_data\\maps_fixed_kernel','images'))
    print("len(img_paths_train) =", len(img_paths_train))
    img_paths_test = []
    for img_path in glob.glob(os.path.join(paths_test, '*.h5')):
        img_paths_test.append(str(img_path).replace('.h5', '.jpg').replace('test_data\\maps_fixed_kernel', 'images'))
    print("len(img_paths_test) =", len(img_paths_test))

    random.shuffle(img_paths_train)
    lst_to_write = [img_paths_train, img_paths_train[:int(len(img_paths_train)*validation_split)], img_paths_test]
    for idx, i in enumerate(['train', 'val', 'test']):
        with open(root+'/paths_train_val_test/paths_'+i+'.txt', 'w') as fout:
            fout.write(str(lst_to_write[idx]))
    return None


def ssim_loss(y_true, y_pred, c1=0.01**2, c2=0.03**2):
    # Generate a 11x11 Gaussian kernel with standard deviation of 1.5
    weights_initial = np.multiply(
        cv2.getGaussianKernel(11, 1.5),
        cv2.getGaussianKernel(11, 1.5).T
    )
    weights_initial = weights_initial.reshape(*weights_initial.shape, 1, 1)
    weights_initial = K.cast(weights_initial, tf.float32)

    mu_F = tf.nn.conv2d(y_pred, weights_initial, [1, 1, 1, 1], padding='SAME')
    mu_Y = tf.nn.conv2d(y_true, weights_initial, [1, 1, 1, 1], padding='SAME')
    mu_F_mu_Y = tf.multiply(mu_F, mu_Y)
    mu_F_squared = tf.multiply(mu_F, mu_F)
    mu_Y_squared = tf.multiply(mu_Y, mu_Y)

    sigma_F_squared = tf.nn.conv2d(tf.multiply(y_pred, y_pred), weights_initial, [1, 1, 1, 1], padding='SAME') - mu_F_squared
    sigma_Y_squared = tf.nn.conv2d(tf.multiply(y_true, y_true), weights_initial, [1, 1, 1, 1], padding='SAME') - mu_Y_squared
    sigma_F_Y = tf.nn.conv2d(tf.multiply(y_pred, y_true), weights_initial, [1, 1, 1, 1], padding='SAME') - mu_F_mu_Y

    ssim = ((2 * mu_F_mu_Y + c1) * (2 * sigma_F_Y + c2)) / ((mu_F_squared + mu_Y_squared + c1) * (sigma_F_squared + sigma_Y_squared + c2))

    return 1 - tf.reduce_mean(ssim, reduction_indices=[1, 2, 3])


def ssim_eucli_loss(y_true, y_pred, alpha=0.001):
    ssim = ssim_loss(y_true, y_pred)        # 一种正则项方法，多用于比较两幅图像的相似度
    eucli = mean_squared_error(y_true, y_pred)
    loss = eucli + alpha * ssim
    # loss = eucli
    return loss


if __name__ == '__main__':
    eval_path_files(root="O:/dataset/crowd/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/pre_model")


