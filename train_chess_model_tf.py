import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dropout, Dense, Add, Activation
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger


PATH_OF_DATASET = "train_750.npz"
PATH_OF_BEST_WEIGHTS = "best_weights.h5"
PATH_OF_CSV_LOGGER = "result_of_fit.csv"


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


def get_tf_resnet_chess_model() -> tf.keras.Model:
    """
    Создает модель с заданной архитектурой.

    :return: TF-модель.
    """
    inputs = Input(shape=(14, 8, 8))
    x = Conv2D(filters=56, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x_skip = x
    x = Conv2D(filters=56, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_skip])
    x = Activation('relu')(x)

    x_skip = x
    x = Conv2D(filters=56, kernel_size=7, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_skip])
    x = Activation('relu')(x)

    x_skip = x
    x = Conv2D(filters=56, kernel_size=7, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_skip])
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(28, activation='relu')(x)
    outputs = Dense(1, activation='tanh')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    model = get_tf_resnet_chess_model()
    model.compile(optimizer=Adadelta(), loss='mean_squared_error')
    model.summary()

    checkpoint = ModelCheckpoint(PATH_OF_BEST_WEIGHTS, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    csv_logger = CSVLogger(PATH_OF_CSV_LOGGER)

    dataset = ChessDataset(PATH_OF_DATASET)

    model.fit(dataset.inputs, dataset.targets, batch_size=1024, epochs=500, shuffle=True,
              verbose=1, validation_split=0.05, callbacks=[checkpoint, csv_logger])
