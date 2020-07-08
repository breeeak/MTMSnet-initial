from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,LambdaCallback

from some_utils import *
from MTMSNet import MTMSNet


def train(root_path='H:/label/fabric_for_label', modelnames="_MTMSnet5_BCE200_con_test"):
    # ############## #
    # #####初始化### #
    # ############## #
    img_paths_test, img_paths_train, img_paths_val = gen_paths(root_path)
    img_paths_train = list(set(img_paths_train) - set(img_paths_val))
    x_val, y_val, img_paths_val = gen_x_y(img_paths_val)
    weights_dir = 'models'
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    loss_dir = './models/loss'
    if not os.path.exists(loss_dir):
        os.mkdir(loss_dir)
    path_val_display = img_paths_val[0]
    x_val_display = load_img(path_val_display)
    if path_val_display.endswith(".jpg"):
        h5_path = path_val_display.replace('.jpg', 'FPALL.h5')
    elif path_val_display.endswith(".png"):
        h5_path = path_val_display.replace('.png', 'FPALL.h5')
    elif path_val_display.endswith(".bmp"):
        h5_path = path_val_display.replace('.bmp', 'FPALL.h5')
    y_val_display1 = img_from_h5(h5_path, labeltypes="warp")
    y_val_display2 = img_from_h5(h5_path, labeltypes="weft")
    y_val_display3 = img_from_h5(h5_path, labeltypes="point_warp")
    y_val_display4 = img_from_h5(h5_path, labeltypes="point_weft")
    y_val_display = np.dstack((y_val_display1, y_val_display2, y_val_display3, y_val_display4))
    # ############## #
    # #####定义模型### #
    # ############## #
    model = MTMSNet()
    weights_path = "./models/MSWeights_"+modelnames+".hdf5"
    model.summary()
    optimizer = Adam(lr=1e-5)
    model.compile(optimizer=optimizer, loss=[ssim_eucli_loss, ssim_eucli_loss], metrics=['accuracy'])  # 使用的是均方差加相似度。

    # ############## #
    # #####相关回调函数### #
    # ############## #
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')  # 检查点
    early = EarlyStopping(monitor="val_loss", mode="min", patience=500, verbose=1)      # 提前终止
    redonplat = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=20, verbose=1)       # 缩减学习率
    show_samples_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: plot_pridict(epoch,logs,model, path_val_display,  x_val_display,y_val_display=y_val_display))
    # tesor = TensorBoard(log_dir=loss_dir, histogram_freq=20, write_graph=True, write_images=True)
    callbacks_list = [checkpoint, early, redonplat, show_samples_callback]  # early

    # ############## #
    # #####训练模型### #
    # ############## #
    history = model.fit_generator(generator_x_y(img_paths_train), validation_data=(generator_x_y(img_paths_val)), steps_per_epoch=len(img_paths_train)*4,
                                  validation_steps=len(img_paths_val)*4, epochs=50, verbose=2, callbacks=callbacks_list)

if __name__ == '__main__':
    train('C:/Users/Administrator/Desktop/newTest/test')

