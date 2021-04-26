import os
import time
import uuid

import pandas as pd
import torch

# Чтобы увидеть прогрессбар в PyCharm нужно сделать:
# 'Run' --> 'Edit Configurations...' --> ✔'Emulate terminal in output console'✔
from progress.bar import IncrementalBar


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUMBER_OF_EPOCHS = 1_000
RESULT_FOLDER = os.path.join(os.getcwd(), 'result')

DATASET_PATH = os.path.join(os.getcwd(), 'NORM_FINAL_DATASET.csv')
DATASET_LENGTH = 4_600_000
DATASET_FOR_TRAIN_LENGTH = 4_500_000
CHUNK_SIZE = 50_000

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

    def __init__(self, deep_depth: int) -> None:
        """
        Инициализация модели.

        :param deep_depth: Глубина модели (число слоёв).
        """
        super(ChessModel, self).__init__()

        self.deep_depth = deep_depth

        self.conv_layers = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, (3, 3), padding=(1, 1)), torch.nn.ReLU())]
            + [torch.nn.Sequential(torch.nn.Conv2d(32, 32, (3, 3), padding=(1, 1)),
                                   torch.nn.ReLU()) for _ in range(self.deep_depth - 1)])

        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(2048, 64)
        self.relu = torch.nn.ReLU()
        self.final_linear = torch.nn.Linear(64, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Преобразование входных данных моделью.

        :param x: Входные данные.
        :return: Выходные данные (результат работы модели).
        """
        new_x = torch.zeros(16, 8, 8).to(DEVICE)
        new_x[:14, :, :] = x
        x = new_x
        x.unsqueeze_(0)

        for conv_i in self.conv_layers:
            x = conv_i(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.final_linear(x)
        output = self.sigmoid(x)
        return output


if __name__ == '__main__':
    if not os.path.exists(RESULT_FOLDER):
        os.mkdir(RESULT_FOLDER)
    else:
        raise FileExistsError("Модель уже обучена.")

    if not os.path.exists(DATASET_PATH):
        raise FileExistsError("Датасет не найден.")

    LOG_FILE.write(f"START TIME: {time.asctime()}.\n")

    DATASET_FOR_TRAIN = ChessDataset(DATASET_PATH, start=0, stop=DATASET_FOR_TRAIN_LENGTH)
    DATASET_FOR_CHECK = ChessDataset(DATASET_PATH, start=DATASET_FOR_TRAIN_LENGTH, stop=DATASET_LENGTH)

    MODEL = ChessModel(deep_depth=4).to(DEVICE)
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=0.0005)
    MSE_LOSS = torch.nn.MSELoss()

    for epoch_number in range(1, NUMBER_OF_EPOCHS + 1):
        epoch_bar = IncrementalBar(f'Epoch {epoch_number}/{NUMBER_OF_EPOCHS}:', max=DATASET_FOR_TRAIN_LENGTH)

        # 1.Обучение
        MODEL.train()
        for input_data, target in DATASET_FOR_TRAIN:
            OPTIMIZER.zero_grad()

            output_data = MODEL(input_data)

            loss = MSE_LOSS(output_data, target)
            loss.backward()
            OPTIMIZER.step()
            epoch_bar.next()
        epoch_bar.finish()

        # 2.Проверка
        targets_as_list = list()
        outputs_as_list = list()

        MODEL.eval()
        with torch.no_grad():
            for input_data, target in DATASET_FOR_CHECK:
                output_data = MODEL(input_data)

                targets_as_list.append(float(target[0][0]))
                outputs_as_list.append(float(output_data[0][0]))

        targets_as_tensor = torch.FloatTensor(targets_as_list)
        targets_as_tensor = targets_as_tensor.to(torch.float32)

        outputs_as_tensor = torch.FloatTensor(outputs_as_list)
        outputs_as_tensor = outputs_as_tensor.to(torch.float32)

        epoch_loss = MSE_LOSS(outputs_as_tensor, targets_as_tensor)

        print(f"Epoch {epoch_number}/{NUMBER_OF_EPOCHS}: MSE = {epoch_loss}.")
        LOG_FILE.write(f"[{time.asctime()}]: Epoch {epoch_number}/{NUMBER_OF_EPOCHS}: MSE = {epoch_loss}.\n")
        LOG_FILE.flush()
        os.fsync(LOG_FILE)

        # 3.Сохранение
        torch.save(MODEL.state_dict(), os.path.join(RESULT_FOLDER, f'model_epoch_{epoch_number}.pt'))

    LOG_FILE.write(f"STOP TIME: {time.asctime()}.\n")


LOG_FILE.close()
