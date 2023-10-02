from pettingzoo import AECEnv
from env.agent import Player
# from env.card import Card
# from env.card import PlayingCard, TaskCards
import unittest
import random
from collections import Counter, defaultdict
REWARD_MAP = {
    "win": 10,
    "lose": -10,
    "task_complete": 1,
}
def env(**kwargs):
    return raw_env(**kwargs)


class raw_env(AECEnv):
    def __init__(
        self, colors: int = 4, ranks: int = 9, rockets: int = 4, players: int = 3, tasks: int = 3
    ):
        self.__config = {
            "colors": colors,
            "ranks": ranks,
            "rockets": rockets,
            "players": players,
            "tasks": tasks,
        }
        self.reset()

    def reset(self):
        # assume play order is player_1 -> player_2 -> player_3 -> ... -> player_n -> player_1 ...
        self.possible_agents = [f"player_{i}" for i in range(self.__config["players"])]
        self.playing_cards, self.task_cards = self.__generateAllCards()
        random.shuffle(self.playing_cards)
        random.shuffle(self.task_cards)        
        self.hands = []
        for _ in range(len(self.possible_agents)):
            self.hands.append([])
        self.tasks = []
        for _ in range(len(self.possible_agents)):
            self.tasks.append([])
        self.commander = -1
        start_idx = 0
        self.task_dict = {}

        #count of cards in each suit for each player. Useful for checking legal plays
        self.suit_counts = [Counter() for _ in range(len(self.possible_agents))]
        for card in self.playing_cards:
            if card[0] == "R" and card[1] == self.__config["rockets"]:
                self.commander = start_idx
            self.hands[start_idx].append(card)
            self.suit_counts[start_idx][card[0]] += 1
            start_idx = (start_idx + 1) % len(self.possible_agents)
        for hand in self.hands:
            hand.sort()
            
        if self.__config["rockets"] == 0:
            self.commander = random.randint(0, len(self.possible_agents) - 1)


        # self.commander = 0
        start_idx = self.commander
        for _ in range(self.__config["tasks"]):
            task = self.task_cards.pop(0)
            self.tasks[start_idx].append(task)
            self.task_dict[task] = start_idx
            start_idx = (start_idx + 1) % len(self.possible_agents)
        self.player_to_play = self.commander
        self.current_trick: list[tuple[Player, tuple[str,int]]] = [] 
        
    # Perfect information function. All hands returned. 
    # Will need to be updated to return only the hand of the current player, as well as belief distribution over the other hands (?).
    # Also, note that we'll need to track the previously played cards as well.
    def observe(self, agent):
        return self.hands, self.tasks, self.current_trick, self.commander, self.suit_counts, self.player_to_play, self.task_dict


    # for now, action is just the index of the card to play. Hints will be added later.
    # Return value is false if the game is over, true otherwise.
    def step(self, action: int) -> int:
        '''
        action: index of card to play
        
        Returns:
            int: 1 if win, -1 if loss, 0 otherwise
        '''
        if action >= len(self.hands[self.player_to_play]):
            raise ValueError(f"Invalid action, value {action} out of range.")
        
        card = self.hands[self.player_to_play].pop(action)
        self.suit_counts[self.player_to_play][card[0]] -= 1
    
        # check if card playable in current trick
        if len(self.current_trick) > 0:
            if self.current_trick[0][1][0] != card[0] and self.suit_counts[self.player_to_play][self.current_trick[0][1][0]] > 0:
                raise ValueError(f"Invalid action, card {card} not playable. Trick so far: {self.current_trick}")

        self.current_trick.append((self.player_to_play, card))

        # check if trick is over
        if len(self.current_trick) == len(self.possible_agents):
            trick_suit = self.current_trick[0][1][0]
            trick_value = self.current_trick[0][1][1]
            winner = self.current_trick[0][0]
            for player, card in self.current_trick:
                if card[0] == trick_suit and card[1] > trick_value:
                    trick_value = card[1]
                    winner = player
                elif card[0] == "R" and trick_suit != "R":
                    trick_value = card[1]
                    trick_suit = "R"
                    winner = player
            
            # check tasks
            for _, card in self.current_trick:
                if card in self.task_dict:
                    player = self.task_dict[card]
                    if player == winner:
                        self.tasks[player].remove(card)
                        self.tasks[player].sort()
                        self.task_dict.pop(card)
                        # TODO: insert code to update reward positively.
                        if sum(len(t) for t in self.tasks) == 0:
                            return 1
                    else:
                        # TODO: insert code to update reward negatively
                        return -1
            self.player_to_play = winner
            self.current_trick = []
        else:
            self.player_to_play = (self.player_to_play + 1) % len(self.possible_agents)
        return 0

    def render(self):
        s = f"config: {self.__config} \n"
        s+= f"hands: {self.hands} \n"
        s+= f"tasks: {self.task_dict} \n"
        s+= f"commander: {self.commander}\n"
        s+= f"current_trick: {self.current_trick}\n"
        s+= f"current_turn: {self.player_to_play}\n"
        return s

    def config(self):
        print(self.__config)
        print(self.possible_agents)
        print(self.playing_cards)
        print(self.task_cards)

    def __generateAllCards(self):
        playing_cards = []
        task_cards = []
        # Color Cards & Task Cards
        for color in ["B", "P", "G", "Y"][: self.__config["colors"]]:
            for rank in range(1, self.__config["ranks"] + 1):
                playing_cards.append((color, rank))
                task_cards.append((color, rank))
        # Rocket Cards
        for value in range(1, self.__config["rockets"] + 1):
            playing_cards.append(("R", value))
        return playing_cards, task_cards


