from env.thecrew_env import env
from env.agent import Player
from env.card import Card
import random


def run_game(strategy):
    my_env = env(colors=3, ranks=1, rockets=0, players=3, tasks=1)
    wins = 0
    losses = 0
    for seed in range(1):
        random.seed(seed)
        my_env.reset()
        print(my_env.render())

        ret_val = 0
        while ret_val == 0:
            hands, tasks, current_trick, commander, suit_counts, player_to_play, task_dict = my_env.observe(0) # observe doesn't vary by agent right now
            legal_moves = get_legal_moves(hands, current_trick, suit_counts, player_to_play)
            print(legal_moves)
            action = strategy(hands, task_dict, current_trick, commander, suit_counts, player_to_play, legal_moves)
            print(f"Player {player_to_play} plays {action}")
            ret_val = my_env.step(hands[player_to_play].index(action))
            print(my_env.render())
        print(f"Game over. Winner: {ret_val}")
        if ret_val == 1:
            wins += 1
        else:
            losses += 1
    print(f"Wins: {wins}, Losses: {losses}")

def random_strategy(hands, task_dict, current_trick, commander, suit_counts, player_to_play, legal_moves):
    return random.choice(legal_moves)


# this strategy involves playing tasks that can be won this trick, otherwise playing randomly
def simple_baseline(hands, task_dict, current_trick, commander, suit_counts, player_to_play, legal_moves):
    # identify current top card in trick

    if len(current_trick) > 0:
        # identify current top card in trick
        winner = current_trick[0][0]
        trick_suit = current_trick[0][1][0]
        trick_value = current_trick[0][1][1]
        for player, card in current_trick:
            if card[0] == trick_suit and card[1] > trick_value:
                trick_value = card[1]
                winner = player
            elif card[0] == "R" and trick_suit != "R":
                trick_value = card[1]
                trick_suit = "R"
                winner = player


        # identify player who needs to win any tasks currently in trick
        needs_to_win = None

        for _, card in current_trick:
            # if multiple tasks in the trick, only focus on one. Doesn't matter much
            if card in task_dict:

                #this player needs to win this trick
                needs_to_win = task_dict[card]
                
                if needs_to_win == player_to_play:
                    # play highest card within trick
                    if legal_moves[0][0] == trick_suit:
                        return max(legal_moves, key=lambda x: x[1])
                    rockets = [c for c in hands[player_to_play] if c[0] == "R"]
                    
                    # if none, play highest rocket
                    if len(rockets) > 0:
                        return max(rockets, key=lambda x: x[1])

                    # play whatever. We're boned
                    return random.choice(legal_moves)

                # player has already played a card, 
                if needs_to_win in [c[0] for c in current_trick]:
                    break
                needs_to_win_playable_cards = [c for c in hands[needs_to_win] if c[0] == trick_suit]
                
                # maybe can win with a rocket
                if len(needs_to_win_playable_cards) == 0:
                    needs_to_win_playable_cards = hands[c for c in hands[needs_to_win] if c[0] == "R"] 
                
                # screw it. Just try to win another mission, but game will end this trick
                if len(needs_to_win_playable_cards) == 0:
                    break
                



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