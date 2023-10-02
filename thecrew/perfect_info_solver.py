from env.thecrew_env import env
from env.agent import Player
from env.card import Card
import random


def run_game(strategy):
    my_env = env(colors=4, ranks=3, rockets=0, players=3, tasks=1)
    wins = 0
    losses = 0
    for seed in range(10000):
        random.seed(seed)
        my_env.reset()
        print(my_env.render())

        ret_val = 0
        while ret_val == 0:
            hands, tasks, current_trick, commander, suit_counts, player_to_play, task_dict = my_env.observe(0) # observe doesn't vary by agent right now
            legal_moves = get_legal_moves(hands, current_trick, suit_counts, player_to_play)
            print(legal_moves)
            action = strategy(hands, tasks, current_trick, commander, suit_counts, player_to_play, legal_moves)
            print(f"Player {player_to_play} plays {action}")
            ret_val = my_env.step(hands[player_to_play].index(action))
            print(my_env.render())
        print(f"Game over. Winner: {ret_val}")
        if ret_val == 1:
            wins += 1
        else:
            losses += 1
    print(f"Wins: {wins}, Losses: {losses}")

def random_strategy(hands, tasks, current_trick, commander, suit_counts, player_to_play, legal_moves):
    return random.choice(legal_moves)


# this strategy involves playing tasks that can be won this trick, otherwise playing randomly
def simple_baseline(hands, tasks, current_trick, commander, suit_counts, player_to_play, legal_moves):
    for player, p_tasks in tasks:
        for task in p_tasks:
            pass


def get_legal_moves(hands, current_trick, suit_counts, player_to_play):
    # if first card in trick, can play anything
    if len(current_trick) == 0:
        return hands[player_to_play]
    # if not first card in trick, must play same suit if possible
    else:
        suit = current_trick[0][1][0]
        if suit_counts[player_to_play][suit] > 0:
            return [card for card in hands[player_to_play] if card[0] == suit]
        else:
            return hands[player_to_play]
        

if __name__ == "__main__":
    random.seed(0)
    run_game(random_strategy)