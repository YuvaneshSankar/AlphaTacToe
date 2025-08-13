import random

def check_winner(board):
    # Rows
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] != 0:
            return board[row][0] 
    # Cols
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != 0:
            return board[0][col] 
    # Diagonals
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return board[0][0] 
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return board[0][2]
    return 0

def check_legal_move(board, row, col):
    return 0 <= row < 3 and 0 <= col < 3 and board[row][col] == 0

def mutate_state(board, row, col, play):
    if check_legal_move(board, row, col):
        new_board = [r[:] for r in board] 
        new_board[row][col] = play
        return new_board
    else:
        raise ValueError("Illegal move")

def is_full(board):
    return all(cell != 0 for row in board for cell in row)


def dfs(board, player):
    global non_terminal_states, x_wins, y_wins, draws, terminal_states, visited

    if board in visited:
        return
    visited.append([r[:] for r in board])

    winner = check_winner(board)
    if winner != 0:
        terminal_states.append([r[:] for r in board])
        if winner == 1:
            x_wins += 1
        else:
            y_wins += 1
        return

    if is_full(board):
        terminal_states.append([r[:] for r in board])
        draws += 1
        return

    non_terminal_states.append([r[:] for r in board])

    for r, c in [(i, j) for i in range(3) for j in range(3) if check_legal_move(board, i, j)]:
        new_board = mutate_state(board, r, c, player)
        dfs(new_board, -player)

    

board = [[0,0,0],
         [0,0,0],
         [0,0,0]]

# 1 -> X, -1 -> O, 0 -> empty

terminal_states=[]
non_terminal_states = []
visited = []

x_wins=0
y_wins=0
draws=0
player=1
dfs(board, player)
print("Total states explored:", len(visited))
print("Total non terminal states:", len(non_terminal_states))
print("Total terminal states:", len(terminal_states))
print("X wins:", x_wins)
print("Y wins:", y_wins)
print("Draws:", draws)

    
# for i in non_terminal_states:
#     for row in i:
#         print(row)
#     print()  # Print a newline for better readability between states
    