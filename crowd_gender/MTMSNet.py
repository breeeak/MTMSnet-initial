from keras.layers import Conv2D, MaxPooling2D, concatenate, Input, Conv2DTranspose, ReLU
from keras.models import Model


def conv_module(x, num_channel_out, kernel_sizes=(2, 2), padding='same', strides=(1, 1), transpose=False):
    if transpose:
        x = Conv2DTranspose(num_channel_out, kernel_sizes, strides=strides, padding=padding, use_bias=True)(x)
    else:
        x = Conv2D(num_channel_out, kernel_sizes, strides=strides, padding=padding, use_bias=True)(x)
    x = ReLU()(x)
    return x


def conv_parallel(x, num_channel_out, reduction_layer=True):
    branches = []
    kernel_sizes = [1, 3, 5, 7]
    num_kernel_sizes = len(kernel_sizes)
    for idx in range(num_kernel_sizes):
        if reduction_layer and idx > 0:
            branch = conv_module(x, num_channel_out*2, (1, 1))
            branch = conv_module(branch, num_channel_out, (kernel_sizes[idx], kernel_sizes[idx]))
        else:
            branch = conv_module(x, num_channel_out, (kernel_sizes[idx], kernel_sizes[idx]))
        branches.append(branch)
    x = concatenate(branches)
    return x


def encoder(x, num_channel_out_lst=[16, 32, 64, 32]):
    for idx_num_channel_out in range(len(num_channel_out_lst)):
        x = conv_parallel(x, num_channel_out_lst[idx_num_channel_out], reduction_layer=(idx_num_channel_out != 0))
        if idx_num_channel_out < len(num_channel_out_lst) - 1:
            x = MaxPooling2D()(x)
    return x


def decoder(x):
    kernel_sizes = [7, 5, 3]
    num_channel_out_lst = [64, 32, 16]
    for idx in range(len(kernel_sizes)):
        x = conv_module(x, num_channel_out_lst[idx], (kernel_sizes[idx], kernel_sizes[idx]))
        x = conv_module(x, num_channel_out_lst[idx], (2, 2), strides=(2, 2), transpose=True)
    x = conv_module(x, 2, (1, 1))
    return x


def decoder2(x):
    kernel_sizes = [7, 5, 3]
    num_channel_out_lst = [64, 32, 16]
    for idx in range(len(kernel_sizes)):
        x = conv_module(x, num_channel_out_lst[idx], (kernel_sizes[idx], kernel_sizes[idx]))
        x = conv_module(x, num_channel_out_lst[idx], (2, 2), strides=(2, 2), transpose=True)
    x = Conv2D(2, (1, 1), activation='sigmoid', strides=(1, 1), padding='same')(x)
    return x


def MTMSNet(input_shape=(None, None, 3)):
    input_flow = Input(input_shape)
    x = encoder(input_flow)
    x_density = decoder(x)
    x_pattern = decoder2(x)
    x_out = [x_density, x_pattern]
    model = Model(inputs=input_flow, outputs=x_out)
    return model
