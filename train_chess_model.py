import os
import time
import uuid

import pandas as pd
import torch

# Чтобы увидеть прогрессбар в PyCharm нужно сделать:
# 'Run' --> 'Edit Configurations...' --> ✔'Emulate terminal in output console'✔
from progress.bar import IncrementalBar


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUMBER_OF_EPOCHS = 300
DIR_FOR_WEIGHTS = os.path.join(os.getcwd(), 'weights')

DATASET_PATH = os.path.join(os.getcwd(), 'TANH_NORM_CHESS_DATASET.csv')
TRAIN_DATASET_LENGTH = 4_500_000
TEST_DATASET_LENGTH = 100_000
CHUNK_SIZE = 500_000

LOG_FILE = open(f'train_{uuid.uuid4()}.log', 'w')


class ChessDataset:
    """
    Датасет для обучения модели игре в шахматы.
    """
    _chunk_size = CHUNK_SIZE

    def __init__(self, path: str, start: int, stop: int) -> None:
        """
        Инициализация датасета.

        Следующее условия должно выполняться для корректной работы:
        (end_index - start_index) % self.chunk_size == 0.

        :param path: Путь к CSV-файлу.
        :param start: Индекс начала нужных данных.
        :param stop: Индекс конца нужный данных.
        """
        self._dataset_path = path

        self._start_index = start
        self._stop_index = stop

        if ChessDataset._chunk_size > self._stop_index - self._start_index:
            self._chunk_size = self._stop_index - self._start_index

    def __len__(self) -> int:
        """
        Возвращает длину нужных данных.

        :return: Длина.
        """
        return self._stop_index - self._start_index

    def __iter__(self) -> (torch.FloatTensor, torch.FloatTensor):
        """
        Реализация метода, поддерживающего цикл for.

        Включает в себя загрузку данных чанками, а не циликом.
        Данные преобразуются в tensor'ы, записываются в память GPU (используя CUDA).

        :return: Тензор входных данных, тензор выходных данных.
        """
        with pd.read_csv(self._dataset_path, sep='|', chunksize=self._chunk_size) as chunk_iter:
            index = -1
            for chunk in chunk_iter:
                if index < self._start_index - 1:
                    index += self._chunk_size
                    continue
                for series in chunk.iloc:
                    index += 1
                    if index < self._stop_index:
                        x_as_list = [eval(matrix_as_str) for matrix_as_str in series.array[:-1]]
                        x_as_tensor = torch.FloatTensor(x_as_list)
                        x_as_tensor = x_as_tensor.to(torch.float32)

                        y_as_float = series.array[-1]
                        y_as_tensor = torch.FloatTensor([[y_as_float]])
                        y_as_tensor = y_as_tensor.to(torch.float32)

                        yield x_as_tensor.to(DEVICE), y_as_tensor.to(DEVICE)
                    else:
                        return


class ChessModel(torch.nn.Module):
    """
    Модель для AI, играющего в шахматы.
    """

    def __init__(self) -> None:
        """
        Инициализация модели.
        """
        super(ChessModel, self).__init__()

        self.conv_1 = torch.nn.Sequential(torch.nn.Conv2d(14, 28, kernel_size=(3, 3), padding=(1, 1)),
                                          torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_2 = torch.nn.Sequential(torch.nn.Conv2d(28, 56, kernel_size=(3, 3), padding=(1, 1)),
                                          torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.flatten = torch.nn.Flatten()
        self.drop_out = torch.nn.Dropout()

        self.linear_1 = torch.nn.Linear(in_features=224, out_features=28)
        self.relu = torch.nn.ReLU()

        self.linear_2 = torch.nn.Linear(in_features=28, out_features=1)

        self.tanh = torch.nn.Tanh()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Преобразование входных данных моделью.

        :param x: Входные данные.
        :return: Выходные данные (результат работы модели).
        """
        x.unsqueeze_(0)

        x = self.conv_1(x)
        x = self.conv_2(x)

        x = self.flatten(x)
        x = self.drop_out(x)

        x = self.linear_1(x)
        x = self.relu(x)

        x = self.linear_2(x)

        return self.tanh(x)


if __name__ == '__main__':
    if not os.path.exists(DIR_FOR_WEIGHTS):
        os.mkdir(DIR_FOR_WEIGHTS)
    else:
        raise FileExistsError("Существует папка с весами. Видимо, модель уже обучена.")

    if not os.path.exists(DATASET_PATH):
        raise FileExistsError("Датасет не найден.")

    LOG_FILE.write(f"START TIME: {time.asctime()}.\n")

    DATASET_FOR_TRAIN = ChessDataset(DATASET_PATH, start=0, stop=TRAIN_DATASET_LENGTH)
    DATASET_FOR_TEST = ChessDataset(DATASET_PATH, start=TRAIN_DATASET_LENGTH,
                                    stop=TRAIN_DATASET_LENGTH + TEST_DATASET_LENGTH)

    MODEL = ChessModel().to(DEVICE)
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=0.0005)
    MSE_LOSS = torch.nn.MSELoss()

    test_losses = list()

    for epoch_number in range(1, NUMBER_OF_EPOCHS + 1):
        epoch_bar = IncrementalBar(f'Epoch {epoch_number}/{NUMBER_OF_EPOCHS}:', max=TRAIN_DATASET_LENGTH)

        # 1.Обучение
        epoch_train_loss = 0
        MODEL.train()

        for input_data, target in DATASET_FOR_TRAIN:
            OPTIMIZER.zero_grad()

            output_data = MODEL(input_data)

            train_loss = MSE_LOSS(output_data, target)
            train_loss.backward()
            epoch_train_loss += train_loss

            OPTIMIZER.step()
            epoch_bar.next()

        epoch_bar.finish()
        epoch_train_loss = epoch_train_loss / TRAIN_DATASET_LENGTH

        # 2.Проверка
        epoch_test_loss = 0
        MODEL.eval()

        with torch.no_grad():
            for input_data, target in DATASET_FOR_TEST:
                output_data = MODEL(input_data)
                epoch_test_loss += MSE_LOSS(output_data, target)

        epoch_test_loss = epoch_test_loss / TEST_DATASET_LENGTH

        # 3.Сохранение и вывод результатов.
        epoch_loss_as_str = f"TRAIN_MSE = {epoch_train_loss}, TEST_MSE = {epoch_test_loss}."
        print(f"Epoch {epoch_number}/{NUMBER_OF_EPOCHS}: {epoch_loss_as_str}")
        LOG_FILE.write(f"[{time.asctime()}]: Epoch {epoch_number}/{NUMBER_OF_EPOCHS}: {epoch_loss_as_str}\n")
        LOG_FILE.flush()
        os.fsync(LOG_FILE)

        torch.save(MODEL.state_dict(), os.path.join(DIR_FOR_WEIGHTS, f'model_epoch_{epoch_number}.pt'))

        # 4. Тест на переобучение модели.
        if len(test_losses) > 15:
            counter = 0
            for prev_loss in test_losses[-15:]:
                if prev_loss < epoch_test_loss:
                    counter += 1

            if counter == 15:
                raise Exception('Overfitting has begun!')
            else:
                test_losses.append(epoch_test_loss)

    LOG_FILE.write(f"STOP TIME: {time.asctime()}.\n")


LOG_FILE.close()
