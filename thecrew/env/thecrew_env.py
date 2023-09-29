from pettingzoo import AECEnv
from thecrew.env.agent import Player
from thecrew.env.card import Card
from thecrew.env.card import PlayingCard, TaskCards
import unittest
import random
from collections import Counter
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

    def reset(self, seed=42):
        random.seed(seed)

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

        #count of cards in each suit for each player. Useful for checking legal plays
        self.suit_counts = [Counter() for _ in range(len(self.possible_agents))]
        for card in self.playing_cards:
            if card.suit() == "R" and card.value() == self.__config["rockets"]:
                self.commander = start_idx
            self.hands[start_idx].append(card)
            self.suit_counts[start_idx][card.suit()] += 1
            start_idx = (start_idx + 1) % len(self.possible_agents)
        for hand in self.hands:
            hand.sort()
            
        for _ in range(self.__config["tasks"]):
            self.tasks[start_idx].append(self.task_cards.pop(0))
            start_idx = (start_idx + 1) % len(self.possible_agents)
        self.player_to_play = self.commander
        self.current_trick: list[tuple[Player, Card]] = [] 
        
    # Perfect information function. All hands returned. 
    # Will need to be updated to return only the hand of the current player, as well as belief distribution over the other hands (?).
    # Also, note that we'll need to track the previously played cards as well.
    def observe(self, agent):
        return self.hands, self.tasks, self.current_trick, self.commander, self.suit_counts, self.player_to_play


    # for now, action is just the index of the card to play. Hints will be added later.
    # Return value is false if the game is over, true otherwise.
    def step(self, action: int) -> bool:
        if action >= len(self.hands[self.player_to_play]):
            raise ValueError(f"Invalid action, value {action} out of range.")
        
        card = self.hands[self.player_to_play].pop(action)
        self.suit_counts[self.player_to_play][card.suit()] -= 1
    
        # check if card playable in current trick
        if len(self.current_trick) > 0:
            if self.current_trick[0][1].suit() != card.suit() and self.suit_counts[self.player_to_play][self.current_trick[0][1].suit()] > 0:
                raise ValueError(f"Invalid action, card {card} not playable. Trick so far: {self.current_trick}")

        self.current_trick.append((self.possible_agents[self.player_to_play], card))

        # check if trick is over
        if len(self.current_trick) == len(self.possible_agents):
            trick_suit = self.current_trick[0][1].suit()
            trick_value = self.current_trick[0][1].value()
            leader = self.current_trick[0][0]
            for player, card in self.current_trick:
                if card.suit() == trick_suit and card.value() > trick_value:
                    trick_value = card.value()
                    leader = player
                elif card.suit() == "R" and trick_suit != "R":
                    trick_value = card.value()
                    trick_suit = "R"
                    leader = player
            
            # check tasks
            for player in range(len(self.possible_agents)):
                for task in self.tasks[player]:
                    if task.suit() == trick_suit and task.value() == trick_value:
                        if player == leader:
                            self.tasks[player].remove(task)
                            self.tasks[player].sort()
                            # TODO: insert code to update reward positively.
                            pass
                        else:
                            # TODO: insert code to update reward negatively
                            return False
            self.player_to_play = leader
            self.current_trick = []
        else:
            self.player_to_play = (self.player_to_play + 1) % len(self.possible_agents)
        return True

    def render(self):
        s = f"config: {self.__config} \n"
        s+= f"hands: {self.hands} \n"
        s+= f"tasks: {self.tasks} \n"
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
                playing_cards.append(PlayingCard(color, rank))
                task_cards.append(TaskCards(color, rank))
        # Rocket Cards
        for value in range(1, self.__config["rockets"] + 1):
            playing_cards.append(PlayingCard("R", value))
        return playing_cards, task_cards
