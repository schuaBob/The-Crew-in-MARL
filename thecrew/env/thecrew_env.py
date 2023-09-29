from pettingzoo import AECEnv
from thecrew.env.agent import Player
from thecrew.env.card import Card
from thecrew.env.card import PlayingCard, TaskCards
import unittest
import random
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

        for card in self.playing_cards:
            if card.suit() == "R" and card.value() == self.__config["rockets"]:
                self.commander = start_idx
            self.hands[start_idx].append(card)
            start_idx = (start_idx + 1) % len(self.possible_agents)
            
        for _ in range(self.__config["tasks"]):
            self.tasks[start_idx].append(self.task_cards.pop(0))
            start_idx = (start_idx + 1) % len(self.possible_agents)
        self.current_turn = self.commander
        self.current_trick: list[tuple[Player, Card]] = [] 
        
    # Perfect information function. All hands returned. 
    # Will need to be updated to return only the hand of the current player, as well as belief distribution over the other hands (?).
    # Also, note that we'll need to track the previously played cards as well.
    def observe(self, agent):
        return self.hands, self.tasks, self.current_trick, self.commander

    def step(self, action):
        pass
    
    def render(self):
        s = f"config: {self.__config} \n"
        s+= f"hands: {self.hands} \n"
        s+= f"tasks: {self.tasks} \n"
        s+= f"commander: {self.commander}\n"
        s+= f"current_trick: {self.current_trick}\n"
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
