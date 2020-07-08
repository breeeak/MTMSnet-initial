
from crowd_gender.utils import *
from crowd_gender.MTMSNet import MTMSNet

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,TensorBoard,LambdaCallback



def train(root_path='H:/label/fabric_for_label', modelname="MTMSnet"):
    # ############## #
    # #####初始化### #
    # ############## #
    # 读取路径
    img_paths_test, img_paths_train, img_paths_val = gen_paths(root_path)

    weights_dir = 'models'
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    loss_dir = './models/loss'
    if not os.path.exists(loss_dir):
        os.mkdir(loss_dir)

    #  验证图片
    path_val_display = img_paths_val[0]
    x_val_display = load_img(path_val_display)
    if path_val_display.endswith(".jpg"):
        h5_path = str(path_val_display).replace('.jpg', '.h5').replace('images','train_data\\maps_fixed_kernel')

    y_val_display = img_from_h5(h5_path, labeltypes="density")

    # ############## #
    # #####定义模型### #
    # ############## #
    model = MTMSNet()
    model.summary()
    optimizer = Adam(lr=1e-5)
    model.compile(optimizer=optimizer, loss=ssim_eucli_loss, metrics=['accuracy'])  # 使用的是均方差加相似度。

    strc_path = "./models/strcture/Strc_"+modelname+".json"
    weights_path = "./models/weights/Weights_"+modelname+".hdf5"

    with open(strc_path, 'w') as fout:
        fout.write(model.to_json())  # 模型写入到json中

    # ############## #
    # #####相关回调函数### #
    # ############## #
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')  # 检查点
    early = EarlyStopping(monitor="val_loss", mode="min", patience=500, verbose=1)      # 提前终止
    redonplat = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=20, verbose=1)       # 缩减学习率
    show_samples_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: plot_pridict(epoch,logs,model, path_val_display,  x_val_display,y_val_display=y_val_display))
    callbacks_list = [checkpoint, early, redonplat, show_samples_callback]  # early

    # ############## #
    # #####训练模型### #
    # ############## #
    # history = model.fit(np.array(x_train), np.array(y_train), batch_size=1, epochs=100, verbose=2, validation_data=(np.array(x_val),np.array(y_val)),callbacks=callbacks_list)
    history = model.fit_generator(generator_x_y(img_paths_train), validation_data=(generator_x_y(img_paths_val)), steps_per_epoch=len(img_paths_train)*6,
                                  validation_steps=len(img_paths_val)*6, epochs=50, verbose=2, callbacks=callbacks_list)


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./models/loss/' + modelname + '_loss.jpg')
    plt.show()
    np.savetxt(os.path.join(loss_dir, modelname + '_loss.txt'), history.history['loss'])
    np.savetxt(os.path.join(loss_dir, modelname + '_acc.txt'), history.history['acc'])

if __name__ == '__main__':
    train("O:/dataset/crowd/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/pre_model")

