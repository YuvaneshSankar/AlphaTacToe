import numpy as np

class TicTacToeGame:
    """
    Standard Tic Tac Toe:
        board: 3x3 grid
        players: 1 (X) and -1 (O)
        win condition: 3 in a row
    """

    def __init__(self):
        self.rows = 3
        self.cols = 3
        self.win = 3  # need 3 in a row

    def get_init_board(self):
        # 3x3 board filled with 0 (empty)
        return np.zeros((self.rows, self.cols), dtype=np.int8)

    def get_board_size(self):
        return (self.rows, self.cols)

    def get_action_size(self):
        # 9 possible positions
        return self.rows * self.cols

    def get_next_state(self, board, player, action):
        """
        Play action (0..8), return (new_board, next_player).
        Action is mapped to (row, col).
        """
        b = np.copy(board)
        row, col = divmod(action, self.cols)
        b[row, col] = player
        return (b, -player)

    def has_legal_moves(self, board):
        return np.any(board == 0)

    def get_valid_moves(self, board):
        """
        Return binary mask of legal moves, length = 9.
        """
        valid_moves = [0] * self.get_action_size()
        for r in range(self.rows):
            for c in range(self.cols):
                if board[r, c] == 0:
                    valid_moves[r * self.cols + c] = 1
        return np.array(valid_moves, dtype=np.int8)

    def is_win(self, board, player):
        # Check rows
        for r in range(self.rows):
            if np.all(board[r, :] == player):
                return True
        # Check cols
        for c in range(self.cols):
            if np.all(board[:, c] == player):
                return True
        # Check diagonals
        if np.all(np.diag(board) == player):
            return True
        if np.all(np.diag(np.fliplr(board)) == player):
            return True
        return False

    def get_reward_for_player(self, board, player):
        """
        Return:
            1  -> if given player has won
           -1  -> if opponent has won
            0  -> if draw
            None -> if game not ended yet
        """
        if self.is_win(board, player):
            return 1
        if self.is_win(board, -player):
            return -1
        if self.has_legal_moves(board):
            return None
        return 0  # draw

    def get_canonical_board(self, board, player):
        """
        From player's perspective: multiply by player.
        So current player's pieces become +1, opponent’s become -1.
        """
        return player * board
