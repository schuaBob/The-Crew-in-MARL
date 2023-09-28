from pettingzoo import AECEnv
from thecrew.env.card import PlayingCard, TaskCards
import unittest


def env(**kwargs):
    return raw_env(**kwargs)


class raw_env(AECEnv):
    def __init__(
        self, colors: int = 4, ranks: int = 9, rockets: int = 4, players: int = 3
    ):
        self.__config = {
            "colors": colors,
            "ranks": ranks,
            "rockets": rockets,
            "players": players,
        }
        self.possible_agents = [f"player_{i+1}" for i in range(players)]
        self.playing_cards, self.task_cards = self.__generateAllCards()

    def config(self):
        print(self.possible_agents)
        print(self.playing_cards)
        print(self.task_cards)
        print(self.__config)

    def __generateAllCards(self):
        playing_cards = []
        task_cards = []
        # Color Cards & Task Cards
        for color in ["B", "P", "G", "Y"][: self.__config["colors"]]:
            for rank in range(1, self.__config["ranks"] + 1):
                playing_cards.append(PlayingCard(color, rank))
                task_cards.append(TaskCards(color, rank, None))
        # Rocket Cards
        for value in range(1, self.__config["rockets"] + 1):
            playing_cards.append(PlayingCard("R", value))
        return playing_cards, task_cards
