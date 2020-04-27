import numpy as np
import tarfile
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.utils import plot_model
import tensorflow as tf
from tensorflow_core import python
from IPython.display import clear_output
import cv2
from tensorflow.python.client import device_lib
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

def mean_iou(y_true, y_pred):
    yt0 = y_true[:, :, :, 0]
    yp0 = K.cast(y_pred[:, :, :, 0] > 0.5, 'float32')
    inter = python.count_nonzero(
        tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = python.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1.,
                   tf.cast(inter / union, 'float32'))
    return iou


class fingernailseg:
    epochs = 100
    batch_size = 30
    sz = (320, 320)

    def __init__(self):
        labelpath = 'C:/Users/Administrator/Desktop/raw/all/moving_label'
        maskpath = 'C:/Users/Administrator/Desktop/raw/all/moving_mask'

        label_files = os.listdir(labelpath)
        mask_files = os.listdir(maskpath)

        test_label_path = 'C:/Users/Administrator/Desktop/img/rotate_label'
        test_mask_path = 'C:/Users/Administrator/Desktop/img/rotate_mask'

        test_label_files = os.listdir(test_label_path)
        test_mask_files = os.listdir(test_mask_path)

        # find intersection of two lists
        files = list(set(label_files).intersection(mask_files))
        test_files = list(set(test_label_files).intersection(test_mask_files))
        X_train = []
        y = []

        x_test = []
        y_test = []

        for f in files:
            mask = Image.open(labelpath + '/' + f)
            mask = np.array(mask.resize(self.sz))#192 * 160 * 3
            mask = mask.mean(axis=2)#160,192,3 -> 160,192 RGB取平均值变成单通道
            #mask= [160,192]
            mask[mask < 250] = 0
            mask[mask > 0] = 1
            if not mask.sum():
                continue
            y.append(mask)

            raw = Image.open(maskpath + '/' + f)
            raw = np.array(raw.resize(self.sz))
            X_train.append(raw)
        X_train = np.array(X_train).astype('float32')
        X_train /= 255
        self.X_train = X_train
        y = np.array(y)
        self.y = np.expand_dims(y, 3)
        # self.sz = self.sz[::-1] + (3,)

        for f in test_files:
            mask = Image.open(test_label_path + '/' + f)
            mask = np.array(mask.resize(self.sz))#192 * 160 * 3
            mask = mask.mean(axis=2)#160,192,3 -> 160,192 RGB取平均值变成单通道
            #mask= [160,192]
            mask[mask < 250] = 0
            mask[mask > 0] = 1
            if not mask.sum():
                continue
            y_test.append(mask)

            raw = Image.open(test_mask_path + '/' + f)
            raw = np.array(raw.resize(self.sz))
            x_test.append(raw)

        x_test = np.array(x_test).astype('float32')
        x_test /= 255
        self.x_test = x_test
        y_test = np.array(y_test)
        self.y_test = np.expand_dims(y_test, 3)
        self.sz = self.sz[::-1] + (3,)

        print(X_train.shape)
        print(y.shape)
        print(x_test.shape)
        print(y_test.shape)

    def create_unet(self):
        s = Input(self.sz)
        c1 = Conv2D(8, 3, activation='relu', padding='same')(s)
        c1 = Conv2D(8, 3, activation='relu', padding='same')(c1)
        p1 = MaxPooling2D()(c1)
        c2 = Conv2D(16, 3, activation='relu', padding='same')(p1)
        c2 = Conv2D(16, 3, activation='relu', padding='same')(c2)
        p2 = MaxPooling2D()(c2)
        c3 = Conv2D(32, 3, activation='relu', padding='same')(p2)
        c3 = Conv2D(32, 3, activation='relu', padding='same')(c3)
        p3 = MaxPooling2D()(c3)
        c4 = Conv2D(64, 3, activation='relu', padding='same')(p3)
        c4 = Conv2D(64, 3, activation='relu', padding='same')(c4)
        p4 = MaxPooling2D()(c4)
        c5 = Conv2D(128, 3, activation='relu', padding='same')(p4)
        c5 = Conv2D(128, 3, activation='relu', padding='same')(c5)
        u6 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c5)
        u6 = Concatenate(axis=3)([u6, c4])
        c6 = Conv2D(64, 3, activation='relu', padding='same')(u6)
        c6 = Conv2D(64, 3, activation='relu', padding='same')(c6)
        u7 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(c6)
        u7 = Concatenate(axis=3)([u7, c3])
        c7 = Conv2D(32, 3, activation='relu', padding='same')(u7)
        c7 = Conv2D(32, 3, activation='relu', padding='same')(c7)
        u8 = Conv2DTranspose(16, 2, strides=(2, 2), padding='same')(c7)
        u8 = Concatenate(axis=3)([u8, c2])
        c8 = Conv2D(16, 3, activation='relu', padding='same')(u8)
        c8 = Conv2D(16, 3, activation='relu', padding='same')(c8)
        u9 = Conv2DTranspose(8, 2, strides=(2, 2), padding='same')(c8)
        u9 = Concatenate(axis=3)([u9, c1])
        c9 = Conv2D(8, 3, activation='relu', padding='same')(u9)
        c9 = Conv2D(8, 3, activation='relu', padding='same')(c9)
        outputs = Conv2D(1, 1, activation='sigmoid')(c9)
        self.model = Model(inputs=[s], outputs=[outputs])
        self.model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[mean_iou])

    def build_callbacks(self):
        checkpointer = ModelCheckpoint(filepath='unet4.h5', verbose=0,
                                       save_best_only=True, save_weights_only=True)
        stop_train = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=10, min_lr=0.00001)
        callbacks = [checkpointer, reduce_lr, stop_train, PlotLearning()]
        return callbacks

    def fit(self):
        #validation_data=[self.x_test,self.y_test]
        self.model.fit(self.X_train, self.y, validation_split=0.1, verbose=0,
                       batch_size=self.batch_size, epochs=self.epochs, callbacks=self.build_callbacks())

    def load_model(self):
        self.model.load_weights('unet4.h5')

    @classmethod  # change NN model
    def change_model(cls, net):
        cls.net = net

    @classmethod
    def set_num_epochs(cls, new_epochs):
        cls.epochs = new_epochs

    @classmethod
    def set_batch_size(cls, new_batch_size):
        cls.batch_size = new_batch_size

    @classmethod
    def set_val_split(cls, new_split):
        cls.val_split = new_split

    @classmethod
    def set_sz(cls, new_sz):
        cls.sz = new_sz


# inheritance for training process plot
class PlotLearning(keras.callbacks.Callback, fingernailseg):
    def __init__(self):
        fingernailseg.__init__(self)

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('mean_iou'))
        self.val_acc.append(logs.get('val_mean_iou'))
        self.i += 1
        fig, ax1 = plt.subplots(figsize=(10, 5))
        clear_output(wait=True)

        lns1 = ax1.plot(self.x, self.acc, label="Mean IoU", color='blue', linewidth=3.)
        lns2 = ax1.plot(self.x, self.val_acc, label="Mean validation IoU", color='green', linewidth=3.)

        ax1.set_ylabel('IoU', fontsize=15)
        ax1.set_xlabel('Epoch', fontsize=15)
        ax1.set_title('Fingernails U-Net Segmentation Model')
        ax2 = ax1.twinx()
        lns3 = ax2.plot(self.x, self.losses, '--', label="loss", color='orange')
        lns4 = ax2.plot(self.x, self.val_losses, '--', label="validation loss", color='red')
        ax2.set_ylabel('Loss', fontsize=15)
        lns = lns1 + lns2 + lns3 + lns4
        ax1.legend(lns, [l.get_label() for l in lns])
        ax1.grid(True)
        plt.xlim([-0.05, self.epochs + .05])
        plt.show()
