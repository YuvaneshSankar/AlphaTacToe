board = [[0,0,0],
         [0,0,0],
         [0,0,0]]

# 1 -> X, -1 -> O, 0 -> empty

def print_board(board):
    for row in board:
        print(" | ".join("X" if x == 1 else "O" if x == -1 else " " for x in row))
        print("-" * 9)

def check_winner(board):
    # Check rows
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] != 0:
            return true
    # Check cols
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != 0:
            return true
    # Diagonals
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return true
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return true
    # No winner
    return false


def check_legal_move(board,row,col):
    if row < 0 or row >= 3 or col < 0 or col >= 3:
        if(board[row][col] != 0):
            return False
    return True

def mutate_state(board, row, col, player):
    if check_legal_move(board, row, col):
        board[row][col] = player
    else:
        raise print("Illegal move")
    return board


def is_full(board):
    if check_winner(board):
        return "Game_over"
    else :
        for row in board:
            if 0 in row:
                return False
        return True

def dfs(board,player):
    player=1
    if is_full(board)=="Game_over":
        return 0
    move=[random.randint(0,2),random.randint(0,2)]
    if not check_legal_move(board, move[0], move[1]):
        mutate_state(board, move[0], move[1], player)
        dfs(board, 0)
    else:
        print("Illegal move, trying again")
        