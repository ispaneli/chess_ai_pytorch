from textwrap import wrap
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.engine.sequential import Sequential
import chess


sys.setrecursionlimit(20_000)


def load_model(path_of_weights: str) -> Sequential:
    """
    Возвращает tf-модель с обученными весами.

    :param path_of_weights: Путь к обученным весам.
    :return: Обученная tf-модель.
    """
    model = tf.keras.Sequential()
    model.add(InputLayer(input_shape=(14, 8, 8)))
    model.add(Conv2D(filters=28, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    model.add(Conv2D(filters=56, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    model.add(Flatten())
    model.add(Dropout(rate=0.2))
    model.add(Dense(28, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    model.compile(optimizer=Adam(0.0005), loss='mean_squared_error')
    model.load_weights(path_of_weights)

    return model


CHESS_MODEL = load_model("best_weights.h5")


def _get_one_piece_matrix(board: chess.Board, piece_type: int, color: bool) -> list:
    """
    Преобразовывает ситуацию на доске в матрицу для одного цвета и одной фигуры.

    :param board: Шахматная доска.
    :param piece_type: Тип фигуры.
    :param color: Цвет игрока.
    :return: Матрица.
    """
    board_as_str = str(board.pieces(piece_type, color))
    board_as_str = board_as_str.replace('\n', '').replace(' ', '').replace('.', '0')
    board_as_matrix = [[int(cell) for cell in wrap(line, 1)] for line in wrap(board_as_str, 8)]
    return board_as_matrix


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


def _get_legal_moves_matrix(board: chess.Board, color: bool) -> list:
    """
    Выдает матрицу возможных ходов для одного цвета.

    :param board: Шахматная доска.
    :param color: Цвет игрока.
    :return: Матрица.
    """
    board.turn = color
    legal_moves_as_list = ['0'] * 64

    for move in board.legal_moves:
        board_index = _cell_name_to_matrix_index(move.uci()[2:])
        legal_moves_as_list[board_index] = '1'

    legal_moves_as_str = ''.join(legal_moves_as_list)
    board_as_matrix = [[int(cell) for cell in wrap(line, 1)] for line in wrap(legal_moves_as_str, 8)]
    return board_as_matrix


def board_to_input_data(board: chess.Board) -> np.array:
    """
    Преобразуем шахматную доску в входные для модели np-данные.

    :param board: Шахматная доска.
    :return: Массив numpy.
    """
    matrix = list()

    for piece_type in chess.PIECE_TYPES:
        matrix.append(_get_one_piece_matrix(board, piece_type, chess.WHITE))
        matrix.append(_get_one_piece_matrix(board, piece_type, chess.BLACK))

    matrix.append(_get_legal_moves_matrix(board, chess.WHITE))
    matrix.append(_get_legal_moves_matrix(board, chess.BLACK))

    np_matrix = np.array(matrix, dtype=np.float32)
    return np.expand_dims(np_matrix, 0)


def get_model_scope(board: chess.Board) -> float:
    """
    Выдает результат работы нейронной сети.

    :param board: Шахматная доска.
    :return: Выход нейронной сети.
    """
    np_3d_matrix = board_to_input_data(board)
    result = CHESS_MODEL.predict(np_3d_matrix)
    return result[0][0]


class MinMaxTree:
    """
    Реализация Min-Max дерева решений с альфа-бетта отсечениями.
    """
    # Глубина поиска хода (глубина дерева).
    _depth = 3

    def __init__(self, board: chess.Board) -> None:
        """
        Инициализация объекта класса.

        :param board: Шахматная доска.
        :return: None.
        """
        self._board = board

    def get_best_move(self) -> str:
        """
        Находит лучший ход в данной шахматной ситуации.

        :return: Шахматный ход san.
        """
        best_move = {'score': 1, 'san': NotImplemented}

        for move in self._board.legal_moves:
            self._board.push(move)
            score = self._get_scope_of_move()
            self._board.pop()

            if score < best_move['score']:
                best_move = {'score': score, 'san': move}

        result_as_san = self._board.san(best_move['san'])
        return str(result_as_san)

    def _get_scope_of_move(self, board: chess.Board = NotImplemented, depth: int = NotImplemented,
                           alpha: float = -np.inf, beta: float = np.inf, max_this: bool = False) -> float:
        """
        Выдает оценочное значение для хода.

        :param board: Шахматная доска.
        :param depth: Глубина дерева.
        :param alpha: Параметр ALPHA для отсечения плохих ходов.
        :param beta: Параметр BETA для отсечения плохих ходов.
        :param max_this: Ответ на вопрос: Максимизировать этот слой дерева?
        :return: Оценочное значение хода.
        """
        try:
            if board is NotImplemented:
                board = self._board

            if depth is NotImplemented:
                depth = self._depth - 1
            elif not depth:
                return get_model_scope(board)

            if self._board.is_game_over():
                return get_model_scope(board)

            if max_this:
                max_score = -np.inf

                for move in board.legal_moves:
                    board.push(move)
                    score = self._get_scope_of_move(board, depth - 1, alpha, beta, False)
                    board.pop()

                    max_score = max(max_score, score)
                    alpha = max(alpha, score)

                    if beta <= alpha:
                        break

                return max_score
            else:
                min_score = np.inf

                for move in board.legal_moves:
                    board.push(move)
                    score = self._get_scope_of_move(board, depth - 1, alpha, beta, True)
                    board.pop()

                    min_score = min(min_score, score)
                    beta = min(beta, score)

                    if beta <= alpha:
                        break

                return min_score
        except:
            print(123)






# def find_best_move(board: chess.Board, depth_of_min_max_tree: int, alpha: int, beta: int, max_this: bool):
#     """
#
#     :param board:
#     :param depth_of_min_max_tree:
#     :param alpha:
#     :param beta:
#     :param max_this:
#     :return:
#     """
#
#
# def minimax(board, depth, alpha, beta, maximizing_player):
#   if depth == 0 or board.is_game_over():
#       return get_model_scope(board)
#
#
#   if maximizing_player:
#     max_eval = -np.inf
#     for move in board.legal_moves:
#       board.push(move)
#       eval = minimax(board, depth - 1, alpha, beta, False)
#       board.pop()
#       max_eval = max(max_eval, eval)
#       alpha = max(alpha, eval)
#       if beta <= alpha:
#         break
#     return max_eval
#   else:
#     min_eval = np.inf
#     for move in board.legal_moves:
#       board.push(move)
#       eval = minimax(board, depth - 1, alpha, beta, True)
#       board.pop()
#       min_eval = min(min_eval, eval)
#       beta = min(beta, eval)
#       if beta <= alpha:
#         break
#     return min_eval
#
#
# # this is the actual function that gets the move from the neural network
# def get_best_ai_move(board, depth):
#   min_move = None
#   min_eval = np.inf
#
#   for move in board.legal_moves:
#     board.push(move)
#     eval = minimax(board, depth - 1, -np.inf, np.inf, True)
#     board.pop()
#     if eval < min_eval:
#       min_eval = eval
#       min_move = move
#
#   return str(board.san(min_move))