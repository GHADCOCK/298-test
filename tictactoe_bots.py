import random

def print_board(board):
    """Prints the 3x3 Tic‐Tac‐Toe board."""
    for i in range(3):
        print(board[3*i:3*i+3])
    print()  # Empty line for spacing

def check_winner(board, player):
    """Checks if the given player has a winning line."""
    wins = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
        (0, 4, 8), (2, 4, 6)              # diagonals
    ]
    for a, b, c in wins:
        if board[a] == board[b] == board[c] == player:
            return True
    return False

def get_available_moves(board):
    """Returns a list of board positions (0-8) that are still empty."""
    return [i for i, v in enumerate(board) if v == ' ']

def weighted_choice(choices, weights):
    """Selects one element from 'choices' using the corresponding 'weights'."""
    return random.choices(choices, weights=weights, k=1)[0]

def choose_move(opponent_history, board):
    """
    Chooses a move based on the opponent's past moves.
    
    If there is no history (or none of the available moves have been seen), the bot picks randomly.
    Otherwise, it weights each available move by (frequency + 1) so that moves the opponent
    has used before get a higher chance (this is analogous to a simple prediction in RPS).
    """
    available = get_available_moves(board)
    
    # Calculate the total frequency among available moves.
    total_freq = sum(opponent_history.get(m, 0) for m in available)
    
    if total_freq == 0:
        # No historical bias: choose randomly.
        return random.choice(available)
    else:
        # Weight each available move by its observed frequency (plus 1 for smoothing).
        weights = [opponent_history.get(m, 0) + 1 for m in available]
        return weighted_choice(available, weights)

def play_game():
    """Simulates a Tic-Tac-Toe game between two bot players."""
    board = [' '] * 9  # Representing the board as a list of 9 spaces.
    
    # Each bot will track the opponent's moves.
    # For player X, freq_x tracks moves made by O; for player O, freq_o tracks moves by X.
    freq_x = {}  # Bot X's record of O's moves.
    freq_o = {}  # Bot O's record of X's moves.
    
    current_player = 'X'
    move_count = 0
    
    print("Starting Tic-Tac-Toe game between two bots:\n")
    print_board(board)
    
    while move_count < 9:
        if current_player == 'X':
            # Bot X chooses using the history of O's moves.
            move = choose_move(freq_x, board)
        else:
            # Bot O chooses using the history of X's moves.
            move = choose_move(freq_o, board)
        
        board[move] = current_player
        move_count += 1
        print(f"Player {current_player} chooses position {move}")
        print_board(board)
        
        # After making a move, update the opponent's history:
        # The opponent observes this move, so we add it to their frequency table.
        if current_player == 'X':
            freq_o[move] = freq_o.get(move, 0) + 1
        else:
            freq_x[move] = freq_x.get(move, 0) + 1
        
        # Check for a winner.
        if check_winner(board, current_player):
            print(f"Player {current_player} wins!")
            return
        
        # Switch players.
        current_player = 'O' if current_player == 'X' else 'X'
    
    print("The game is a draw.")

if __name__ == '__main__':
    play_game()
