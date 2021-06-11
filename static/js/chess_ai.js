function randomUUID4() {
    return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
        (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
    );
}

function getSwal(title, text, gradient) {
    swal({
        title: title,
        text: text,
        position: "center",
        backdrop: gradient,
        background: "white",
        allowOutsideClick: true,
        allowEscapeKey: true,
        allowEnterKey: true,
        showConfirmButton: false,
        showCancelButton: false,
        timer: 15000
    });
}

function humanWon() {
    var gradient = "linear-gradient(to top left, rgb(20, 201, 56), rgb(20, 184, 54), rgb(20, 167, 53)," +
        "rgb(21, 151, 51), rgb(21, 134, 50), rgb(21, 117, 48)," +
        "rgb(18, 104, 42), rgb(16, 92, 35), rgb(13, 79, 29)," +
        "rgb(10, 66, 23), rgb(8, 54, 16), rgb(5, 41, 10))";
    getSwal("You've won!", "This time you managed to beat the Chess AI.", gradient);
}

function humanLost() {
    var gradient = "linear-gradient(to top left, rgb(212, 70, 70), rgb(210, 63, 63), rgb(208, 56, 56), rgb(205, 50, 50), rgb(203, 43, 43), rgb(201, 36, 36), rgb(192, 31, 31), rgb(183, 26, 26), rgb(175, 21, 21), rgb(166, 16, 16), rgb(157, 11, 11), rgb(148, 6, 6))";
    getSwal("You've lost!", "This time you failed to defeat Chess AI.", gradient);
}

function draw() {
    var gradient = "linear-gradient(to top left, rgb(199, 176, 0), rgb(210, 168, 0), rgb(221, 160, 0)," +
        "rgb(233, 152, 0), rgb(244, 144, 0), rgb(255, 136, 0)," +
        "rgb(238, 129, 3), rgb(221, 122, 6), rgb(204, 116, 10)," +
        "rgb(187, 109, 13), rgb(170, 102, 16), rgb(153, 95, 19))";
    getSwal("Draw!", "Hmmm... So who is stronger?", gradient);
}

var board, game = new Chess();

var board_uuid4 = randomUUID4();

var onDragStart = function (source, piece, position, orientation) {
    if (game.in_checkmate() === true || game.in_draw() === true ||
        piece.search(/^b/) !== -1) {
        return false;
    }
};

var makeBestMove = function () {
    if (game.game_over()) {
        if (game.in_draw()) {
            draw();
        } else if(game.turn() === 'b') {
            humanWon();
        } else {
            humanLost();
        }
        return;
    }

    var history = game.history();
    console.log(history[history.length - 1]);

    $.ajax({
        type: "POST",
        url: '/new_move',
        data: JSON.stringify({move: history[history.length - 1], uuid4: board_uuid4}),
        contentType: 'application/json',
        success: (data) => {
            game.move(data.move);
            board.position(game.fen());
            renderMoveHistory(game.history());

            if (game.game_over()) {
                if (game.in_draw()) {
                    draw();
                } else if(game.turn() === 'b') {
                    humanWon();
                } else {
                    humanLost();
                }
            }
        },
    });
};


var renderMoveHistory = function (moves) {
    var historyElement = $('#move-history').empty();
    historyElement.empty();

    for (var i = 0; i < moves.length; i = i + 2) {
        historyElement.append('<span>' + moves[i] + ' ' + ( moves[i + 1] ? moves[i + 1] : ' ') + '</span><br>')
    }

    historyElement.scrollTop(historyElement[0].scrollHeight);
};

var onDrop = function (source, target) {
    var move = game.move({
        from: source,
        to: target,
        promotion: 'q'
    });

    removeGreySquares();
    if (move === null) {
        return 'snapback';
    }

    renderMoveHistory(game.history());
    window.setTimeout(makeBestMove, 250);
};

var onSnapEnd = function () {
    board.position(game.fen());
};

var onMouseoverSquare = function(square, piece) {
    var moves = game.moves({
        square: square,
        verbose: true
    });

    if (moves.length === 0) return;

    greySquare(square);

    for (var i = 0; i < moves.length; i++) {
        greySquare(moves[i].to);
    }
};

var onMouseoutSquare = function(square, piece) {
    removeGreySquares();
};

var removeGreySquares = function() {
    $('#board .square-55d63').css('background', '');
};

var greySquare = function(square) {
    var squareEl = $('#board .square-' + square);

    var background = '#a9a9a9';
    if (squareEl.hasClass('black-3c85d') === true) {
        background = '#696969';
    }

    squareEl.css('background', background);
};

var cfg = {
    draggable: true,
    position: 'start',
    onDragStart: onDragStart,
    onDrop: onDrop,
    onMouseoutSquare: onMouseoutSquare,
    onMouseoverSquare: onMouseoverSquare,
    onSnapEnd: onSnapEnd
};

board = ChessBoard('board', cfg);