!yes | pip uninstall python-chess
!pip install progress py-cpuinfo chess

!wget https://raw.githubusercontent.com/Medvate/chess_ai_with_pytorch/stockfish_for_colab/stockfish_13_linux_x64_bmi2

!chmod 755 -R stockfish_13_linux_x64_bmi2

import csv
import os
import sys
import platform
import random
import time
import uuid
from textwrap import wrap
from typing import List
import uuid

import chess
import chess.engine
import cpuinfo
import numpy as np


DATASET_LENGTH = 2_500_000
DATASET_PATH = os.path.join(os.getcwd(), f'drive/MyDrive/chess_datasets/{str(uuid.uuid4()).replace("-", "_")}.csv')
HEADER = ['white_piece_1', 'black_piece_1', 'white_piece_2', 'black_piece_2',
          'white_piece_3', 'black_piece_3', 'white_piece_4', 'black_piece_4',
          'white_piece_5', 'black_piece_5', 'white_piece_6', 'black_piece_6',
          'white_legal_moves', 'black_legal_moves', 'stockfish_white_score']

STOCKFISH_PATH = os.path.join(os.getcwd(), 'stockfish_13_linux_x64_bmi2')
STOCKFISH_ENGINE = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

ENGINE_LIMIT = chess.engine.Limit(depth=16)

MAX_STOCKFISH_SCORE = 10_000
MIN_STOCKFISH_SCORE = -10_000

MU_FOR_RANDOM = 128
SIGMA_FOR_RANDOM = 58


def create_random_board() -> chess.Board:
    """
    Создает случайную ситуацию на шахматной доске.

    Число ходов - это рандомное число нормального распределения,
    описывающего продолжительность реальных игры в шахматы.
    Источник: https://ingram-braun.net/erga/2015/02/on-the-average-move-number-of-a-chess-game/

    :return: Случайная шахматная ситуация.
    """
    random_depth = -1
    while random_depth <= 0:
        random_depth = int(np.random.normal(MU_FOR_RANDOM, SIGMA_FOR_RANDOM, 1))

    random_board = chess.Board()
    for _ in range(random_depth):
        if random_board.is_game_over():
            break

        legal_moves = list(random_board.legal_moves)
        random_move = random.choice(legal_moves)
        random_board.push(random_move)

    return random_board


def _get_stockfish_score(board: chess.Board) -> int:
    """
    Выдает значение, отвечающее на вопрос: "Насколько хорошая позиция сейчас у белых?"

    При вычислении используется Stockfish - лучший шахматный движок по многим рейтингам.

    :param board: Шахматная доска.
    :return: Качество позиции белых.
    """
    result = STOCKFISH_ENGINE.analyse(board, ENGINE_LIMIT, game=str(uuid.uuid4()))
    return result['score'].white().score()


def _str_to_matrix_str(board_as_str: str) -> str:
    """
    Преобразует строку, длиной 64, в матрицу 8x8; затем компактно записывает матрицу,
    как строку (которую можно парсить встроенным методом eval).

    :param board_as_str: Шахматная доска, как строка из 0 и 1.
    :return: Строка-матрица.
    """
    board_as_matrix = [[int(cell) for cell in wrap(line, 1)] for line in wrap(board_as_str, 8)]
    return str(board_as_matrix).replace(' ', '')


def _get_one_piece_board_as_matrix_str(board: chess.Board, piece_type: int, color: bool) -> str:
    """
    Находит строку-матрицу, в которой содержится информация о конкретном типе фигур конретного цвета.

    :param board: Шахматная доска.
    :param piece_type: Тип фигуры, матрицу которой строим (Пешка, Конь, Слон, Ладья, Ферзь или Король).
    :param color: Цвет фигур, для которых считает (Белые или Черные).
    :return: Строка-матрица.
    """
    board_as_str = str(board.pieces(piece_type, color))
    board_as_str = board_as_str.replace('\n', '').replace(' ', '').replace('.', '0')

    return _str_to_matrix_str(board_as_str)


def _cell_name_to_matrix_index(cell_name: str) -> int:
    """
    Преобразуем UCI-название шахматного поля в индекс матрицы.

    Пример: 'f8' --> 5; 'e1' --> 60.

    :param cell_name: Имя клетки по UCI.
    :return: Индекс клетки в матрице.
    """
    letter = cell_name[0]
    number = int(cell_name[1])
    return 8 * (8 - number) + (ord(letter) - 97)


def _get_legal_moves_as_matrix_str(board: chess.Board, color: bool) -> str:
    """
    Находит строку-матрицу, в которой содержатся все возможные ходы фигур выбранного цвета.

    :param board: Шахматная доска.
    :param color: Цвет фигур.
    :return: Строка-матрица.
    """
    board.turn = color

    legal_moves_as_list = ['0' for _ in range(64)]
    for move in board.legal_moves:
        index_of_end_move_cell = _cell_name_to_matrix_index(move.uci()[2:])
        legal_moves_as_list[index_of_end_move_cell] = '1'

    legal_moves_as_str = ''.join(legal_moves_as_list)
    return _str_to_matrix_str(legal_moves_as_str)


def board_to_csv_line(board: chess.Board) -> List[str] or None:
    """
    Преобразует ситуацию на шахматной доске в данные для обучения.

    Иногда Stockfish вместо оценки возвращает None, такие
    шахматные ситуации мы пропускаем (тоже возвращает None).

    :param board: Шахматная доска.
    :return: Список из 14 матриц размерности 8x8, а также оценка Stockfish'а, или None.
    """
    csv_line = list()

    stockfish_scope = _get_stockfish_score(board)

    if stockfish_scope is None:
        return None

    for piece_type in chess.PIECE_TYPES:
        csv_line.append(_get_one_piece_board_as_matrix_str(board, piece_type, chess.WHITE))
        csv_line.append(_get_one_piece_board_as_matrix_str(board, piece_type, chess.BLACK))

    csv_line.append(_get_legal_moves_as_matrix_str(board, chess.WHITE))
    csv_line.append(_get_legal_moves_as_matrix_str(board, chess.BLACK))

    csv_line.append(str(stockfish_scope))

    return csv_line


def create_file_on_google_disk(filepath: str) -> None:
    """
    Создает файл на Google-диске.
    """
    file = open(filepath, 'w')
    file.close()
    print(f'Файл был создан: {filepath}')


if __name__ == '__main__':    
    print(platform.system())
    print(platform.processor())
    print(cpuinfo.get_cpu_info()['brand_raw'])

    from google.colab import drive
    drive.mount('/content/drive')

    create_file_on_google_disk(DATASET_PATH)

    DATASET_WRITER = csv.writer(open(DATASET_PATH, 'w'), delimiter='|')
    DATASET_WRITER.writerow(HEADER)

    start_time = time.time()

    counter = 0
    while counter < DATASET_LENGTH:
        random_board = create_random_board()
        dataset_line = board_to_csv_line(random_board)

        if dataset_line is not None:
            counter += 1
            DATASET_WRITER.writerow(dataset_line)
            sys.stdout.write(f"\r{counter}/{DATASET_LENGTH}")
            sys.stdout.flush()

    delta_time = time.time() - start_time
    print(f"\nПотраченно времени: {int(delta_time // 60)}:{int(delta_time % 60)} мин.")

    STOCKFISH_ENGINE.quit()
