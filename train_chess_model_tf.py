#from abc import ABC

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


class ChessDataset:
    """
    Датасет для обучения модели игре в шахматы.
    """

    def __init__(self, npz_path: str) -> None:
        """
        Инициализация датасета.
        :param npz_path: Путь к CSV-файлу.
        """
        tmp_data = np.load(npz_path)
        self.inputs = tmp_data['inputs']
        self.targets = tmp_data['targets']
        del tmp_data

        self.inputs = np.squeeze(self.inputs, axis=1)
        self.targets = np.squeeze(self.targets, axis=1)

        self.inputs = np.asarray(self.inputs, dtype=np.float32)
        self.targets = np.asarray(self.targets, dtype=np.float32)

        print(f'x.shape={self.inputs.shape}, y.shape={self.targets.shape}.')


# class ChessModel(tf.keras.Model, ABC):
#     def __init__(self):
#         super(ChessModel, self).__init__()
#
#         self.input_layer = Input(shape=(14, 8, 8))
#
#         self.conv_1 = tf.keras.Sequential([Conv2D(filters=28, kernel_size=3, padding='same',
#                                                   activation='relu', data_format='channels_first'),
#                                            MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')])
#         self.conv_2 = tf.keras.Sequential([Conv2D(filters=56, kernel_size=3, padding='same',
#                                                   activation='relu', data_format='channels_first'),
#                                            MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')])
#
#         self.flatten = Flatten()
#         self.dropout = Dropout(rate=0.2)
#         self.dense_1 = Dense(28, activation='relu')
#         self.dense_2 = Dense(1, activation='tanh')
#
#     def call(self, x):
#         x = self.input_layer(x)
#         x = self.conv_1(x)
#         x = self.conv_2(x)
#         x = self.flatten(x)
#         x = self.dropout(x)
#         x = self.dense_1(x)
#         return self.dense_2(x)


# layers_list = [Input(shape=(14, 8, 8)),
#                Conv2D(filters=28, kernel_size=3, padding='same', activation='relu', data_format='channels_first'),
#                MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
#                Conv2D(filters=56, kernel_size=3, padding='same', activation='relu', data_format='channels_first'),
#                MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
#                Flatten(),
#                Dropout(rate=0.2),
#                Dense(28, activation='relu'),
#                Dense(1, activation='tanh')]
#
# model = tf.keras.Sequential(layers_list)
# model = ChessModel()


if __name__ == '__main__':
    model = tf.keras.Sequential()
    model.add(Input(shape=(14, 8, 8)))
    model.add(Conv2D(filters=28, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    model.add(Conv2D(filters=56, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    model.add(Flatten())
    model.add(Dropout(rate=0.2))
    model.add(Dense(28, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    checkpoint = ModelCheckpoint("best_weights.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    model.compile(optimizer=Adam(0.0005), loss='mean_squared_error')
    model.summary()

    dataset = ChessDataset('train.npz')

    model.fit(dataset.inputs, dataset.targets, epochs=300, shuffle=True,
              verbose=1, validation_split=0.05, callbacks=[checkpoint])

    model.save('weights.h5')
