import random
from contextlib import redirect_stdout
# Smoothing factor for move frequencies. Set to 0 for best response
smoothing = 0  
loopCount = 0
deterministic_mode = False  # When True, always choose the highest probability move

def counter_move(opponent_move):
    """
    Returns the move that beats the given opponent move.
    Rock (R) is beaten by Paper (P), 
    Paper (P) is beaten by Scissors (S),
    Scissors (S) is beaten by Rock (R).
    """
    if opponent_move == 'R':
        return 'P'
    elif opponent_move == 'P':
        return 'S'
    elif opponent_move == 'S':
        return 'R'

def choose_move(opponent_history, loopCount):
    """
    Chooses a move based on the opponent's past moves.
    Can operate in deterministic or probabilistic mode.
    """
    moves = ['R', 'P', 'S']
    if not opponent_history:  # No history: pick randomly
        return random.choice(moves)
    
    # Create weights for each move based on opponent's frequencies (with smoothing)
    weights = [opponent_history.get(m, 0) + smoothing for m in moves]
    
    if deterministic_mode:
        # Find the maximum weight
        max_weight = max(weights)
        # Get all indices where weight equals max_weight
        max_indices = [i for i, w in enumerate(weights) if w == max_weight]
        # Randomly choose one of the maximum indices
        chosen_index = random.choice(max_indices)
        predicted_move = moves[chosen_index]
    else:
        # Original probabilistic prediction
        predicted_move = random.choices(moves, weights=weights, k=1)[0]

    if loopCount <= 10:
        for _ in range(12):
            if deterministic_mode:
                max_indices = [i for i, w in enumerate(weights) if w == max(weights)]
                print("Predicted move (deterministic): " + moves[random.choice(max_indices)])
            else:
                print("Predicted move (probabilistic): " + random.choices(moves, weights=weights, k=1)[0])
        print("WEIGHTS: " + str(weights))
    
    return counter_move(predicted_move)

def decide_winner(move1, move2):
    """
    Determines the winner given two moves.
    Returns:
      0 if tie,
      1 if move1 wins,
      2 if move2 wins.
    """
    if move1 == move2:
        return 0
    if (move1 == 'R' and move2 == 'S') or \
       (move1 == 'P' and move2 == 'R') or \
       (move1 == 'S' and move2 == 'P'):
        return 1
    else:
        return 2

def play_game(rounds=100000, deterministic=False):
    """
    Simulates a game of Rock–Paper–Scissors for a fixed number of rounds.
    Both bots update their opponent's move history after each round.
    """
    global deterministic_mode
    deterministic_mode = deterministic

    # Each bot tracks the opponent's move frequencies.
    history_bot1 = {}  # Bot 1 records Bot 2's moves.
    history_bot2 = {}  # Bot 2 records Bot 1's moves.
    
    score_bot1 = 0
    score_bot2 = 0
    loopCount = 0
    
    move_names = {'R': 'Rock', 'P': 'Paper', 'S': 'Scissors'}
    
    print("Starting Rock–Paper–Scissors game between two bots!\n")
    
    for round_num in range(1, rounds + 1):
        # Each bot chooses a move based on the opponent's history.
        move_bot1 = choose_move(history_bot1, loopCount)
        move_bot2 = choose_move(history_bot2, loopCount)
        
        # Determine round outcome.
        result = decide_winner(move_bot1, move_bot2)
        if result == 0:
            outcome = "Tie"
        elif result == 1:
            outcome = "Bot 1 wins"
            score_bot1 += 1
        else:
            outcome = "Bot 2 wins"
            score_bot2 += 1
        loopCount+=1
        # Print the moves and result for this round.
        print(f"Round {round_num}:")
        print(f"  Bot 1 plays {move_names[move_bot1]}")
        print(f"  Bot 2 plays {move_names[move_bot2]}")
        print(f"  Outcome: {outcome}\n")
        if round_num % 1000 == 0:
            print(f"RATIO OF SCORES after {round_num} rounds: {score_bot1}:{score_bot2} = {score_bot1/score_bot2}")
        
        # Update the histories with the moves just played.
        history_bot1[move_bot2] = history_bot1.get(move_bot2, 0) + 1
        history_bot2[move_bot1] = history_bot2.get(move_bot1, 0) + 1
    
    # Print final scores.
    print("Final Score:")
    print(f"  Bot 1: {score_bot1}")
    print(f"  Bot 2: {score_bot2}")
    if score_bot1 == score_bot2:
        print("Overall Result: Tie")
    elif score_bot1 > score_bot2:
        print("Overall Result: Bot 1 wins!")
    else:
        print("Overall Result: Bot 2 wins!")

if __name__ == '__main__':
    # Open a file in write mode and redirect all printed output to this file.
    with open("output.txt", "w") as f:
        with redirect_stdout(f):
            play_game(deterministic=True)  # Set to True to use deterministic mode
