from flask import Flask, render_template, request, json
from chess import Board

from chess_model_tf import MinMaxTree

app = Flask(__name__)

BOARD = Board()
BOARD_UUID4 = 'None'


@app.route('/')
def start_chess_game():
    return render_template('chess.html'), 201


@app.route('/new_move', methods=['POST'])
def get_new_chess_model_move():
    data = json.loads(request.data.decode("utf-8"))

    global BOARD
    global BOARD_UUID4
    if BOARD_UUID4 != data['uuid4']:
        BOARD_UUID4 = data['uuid4']
        BOARD.reset()

    BOARD.push_san(data['move'])
    # result = {'move': get_best_ai_move(BOARD, depth=3)}
    min_max_tree = MinMaxTree(BOARD)
    result = {'move': min_max_tree.get_best_move()}

    BOARD.push_san(result['move'])
    return result


if __name__ == '__main__':
    app.run()
