from gymnasium import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import random
from collections import Counter, deque
from typing import Deque
from bidict import bidict
import numpy as np
import logging
import os
import time

# TODO: need to come up a better reward map, agents can be stupid and still get reward if win for the current reward_map
REWARD_MAP = {
    "win": 50,
    "lose": -50,
    "task_complete": 10,
}


def env(**kwargs):
    return raw_env(**kwargs)


class raw_env(AECEnv):
    def __init__(
        self,
        render: bool,
        colors: int = 4,
        ranks: int = 9,
        rockets: int = 4,
        players: int = 3,
        tasks: int = 3,
    ):
        self.__config = {
            "render": render,
            "colors": colors,
            "ranks": ranks,
            "rockets": rockets,
            "players": players,
            "tasks": tasks,
        }
        self.__setLogger()
        self.metadata = {"name": "The Crew"}
        self.possible_agents: list[str] = [f"player_{i}" for i in range(players)]
        self.__playing_cards, self.__tasks_cards = self.__generateAllCards()
        self.playing_cards_bidict = bidict(
            {k: idx + 1 for idx, k in enumerate(self.__playing_cards)}
        )
        self.action_spaces: dict[str, spaces.Space] = {
            agent: spaces.Discrete(colors * ranks + rockets, start=1)
            for agent in self.possible_agents
        }
        self.observation_spaces: dict[str, spaces.Space] = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=40, shape=(players,), dtype=np.int8
                    ),
                    "action_mask": spaces.MultiBinary(colors * ranks + rockets),
                }
            )
            for agent in self.possible_agents
        }

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self.__config["seed"] = seed
        random.seed(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.__agent_selector = agent_selector(self.agents)
        self.__hands: dict[str, list[tuple[str, int]]] = {}
        self.__suit_counters: dict[str, Counter] = {}
        self.__tasks: dict[str, list[tuple[str, int]]] = {}
        self.__tasks_owner: dict[tuple[str, int], str] = {}
        self.__current_trick: list[tuple[str, tuple[str, int]]] = []
        self.__turn = 0

        for agent in self.agents:
            self.__hands[agent] = []
            self.__suit_counters[agent] = Counter()
            self.__tasks[agent] = []

        self.__deal_playing_cards()

        if self.__config["rockets"] == 0:
            self.__reinit_agents_order(random.choice(self.agents))
        else:
            for agent in self.agents:
                if ("R", self.__config["rockets"]) in self.__hands[agent]:
                    self.__reinit_agents_order(agent)
                    break
        self.__deal_task_cards()
        for agent in self.agents:
            self.__hands[agent].sort()
        self.agent_selection = self.__agent_selector.reset()
        if self.__config["render"]:
            self.config()

    def observe(self, agent):
        observation = np.array(
            [self.playing_cards_bidict[card] for _, card in self.__current_trick],
            dtype=np.int8,
        )
        action_mask = np.zeros(len(self.playing_cards_bidict), dtype=np.int8)
        legal_moves = self.__legal_moves() if agent == self.agent_selection else []
        for i in legal_moves:
            action_mask[i] = 1
        return {
            "observation": observation,
            "action_mask": action_mask,
        }

    def __legal_moves(self):
        """
        Legal Moves of a agent:
        1. if trick_basecard exist, play cards with color same as trick_basecard
        2. if trick_basecard exist, play any card if player don't have cards with color same as the trick_basecard
        3. if trick_basecard not exist(first player in turn), play any cards
        """
        # condition 3 or 2
        if (
            self.__agent_selector.is_first()
            or self.__suit_counters[self.agent_selection][
                (trick_basecard := self.__current_trick[0][1])[0]
            ]
            == 0
        ):
            return [
                self.playing_cards_bidict[card] - 1
                for card in self.__hands[self.agent_selection]
            ]
        # condition 1
        if self.__suit_counters[self.agent_selection][trick_basecard[0]] > 0:
            return [
                self.playing_cards_bidict[card] - 1
                for card in self.__hands[self.agent_selection]
                if card[0] == trick_basecard[0]
            ]

    def __play_turn(self, agent: str, action: int):
        card = self.playing_cards_bidict.inverse[action]
        self.__hands[agent].remove(card)
        self.__suit_counters[agent][card[0]] -= 1
        self.__current_trick.append((agent, card))

    def step(self, action: int):
        """
        action: index of card to play, provided after action_masking and ideally by custom Policy
        """
        # truncation is used when num_step is set
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            if self.__config["render"] and (reason := self.infos[self.agent_selection].get("terminate_reason")):
                self.logger.info(f"Termination Cause: {reason}")
            return self._was_dead_step(action)
        if self.__agent_selector.is_first():
            self.__turn += 1
        self.__play_turn(self.agent_selection, action)
        # check if trick is over
        if self.__agent_selector.is_last():
            trick_suit, trick_value = self.__current_trick[0][1]
            trick_owner = self.__current_trick[0][0]
            for card_player, (card_suit, card_value) in self.__current_trick[1:]:
                if card_suit == trick_suit and card_value > trick_value:
                    trick_value = card_value
                    trick_owner = card_player
                elif card_suit == "R" and trick_suit != "R":
                    trick_suit = "R"
                    trick_value = card_value
                    trick_owner = card_player

            # check if any task is completed
            for card_player, card in self.__current_trick:
                if card in self.__tasks_owner.keys():
                    if (task_owner := self.__tasks_owner[card]) == trick_owner:
                        self.__tasks[task_owner].remove(card)
                        self.__tasks[task_owner].sort()
                        self.__tasks_owner.pop(card)
                        # currently only reward trick_owner(task_owner)
                        for agent in self.agents:
                            self.rewards[agent] += REWARD_MAP["task_complete"]
                    else:
                        self.infos[self.agent_selection][
                            "terminate_reason"
                        ] = f"{task_owner} is unable to fulfill the task {card} due to {trick_owner}'s move of playing {(trick_suit, trick_value)}"
                        # punish the card player of that task
                        self.rewards[card_player] += REWARD_MAP["lose"]
                        for agent in self.agents:
                            self.terminations[agent] = True
                        break
            if len(self.__tasks_owner) == 0:
                self.infos[self.agent_selection][
                    "terminate_reason"
                ] = f"All tasks have completed."
                for agent in self.agents:
                    self.rewards[agent] += REWARD_MAP["win"]
                    self.terminations[agent] = True
        if self.__config["render"]:
            self.render()
        if self.__agent_selector.is_last():
            self.__reinit_agents_order(trick_owner)
            self.__current_trick = []
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self.__agent_selector.next()
        self._accumulate_rewards()

    def render(self):
        if self.__agent_selector.is_first():
            self.logger.info("{:=^40}".format(f" Turn {self.__turn} "))
        card = (
            str(self.__current_trick[-1][1]) + " -> Task Card"
            if self.__current_trick[-1][1] in self.__tasks_owner
            else self.__current_trick[-1][1]
        )
        self.logger.info(f"{self.agent_selection} played {card}")
        if self.__agent_selector.is_last():
            self.logger.info(f"Unfinished Tasks: {self.__tasks_owner}")
            for agent in self.agents:
                self.logger.info(f"{agent} Hand: {self.__hands[agent]}")

    def config(self):
        self.logger.info("{:=^80}".format(f" Parameter Configs "))
        self.logger.info(f"Configs:{self.__config}")
        self.logger.info(f"Agents: {self.possible_agents}")
        for agent in self.agents:
            self.logger.info(f"{agent} Hand: {self.__hands[agent]}")
        self.logger.info(
            f"{self.__agent_selector.agent_order[0]} with ('R', 4) in hand will go first"
        )

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def __generateAllCards(self):
        playing_cards: list[tuple[str, int]] = []
        task_cards: Deque[tuple[str, int]] = deque()
        # Color Cards & Task Cards
        for suit in ["B", "P", "G", "Y"][: self.__config["colors"]]:
            for rank in range(1, self.__config["ranks"] + 1):
                playing_cards.append((suit, rank))
                task_cards.append((suit, rank))
        # Rocket Cards
        for value in range(1, self.__config["rockets"] + 1):
            playing_cards.append(("R", value))
        return playing_cards, task_cards

    def __deal_playing_cards(self):
        random.shuffle(self.__playing_cards)
        self.__agent_selector.reset()
        for card in self.__playing_cards:
            agent = self.__agent_selector.next()
            self.__hands[agent].append(card)
            self.__suit_counters[agent][card[0]] += 1

    def __reinit_agents_order(self, start_agent):
        """Reset order given a start agent"""
        idx = self.agents.index(start_agent)
        new_order = self.agents[idx:] + self.agents[0:idx]
        self.__agent_selector.reinit(new_order)

    def __deal_task_cards(self):
        random.shuffle(self.__tasks_cards)
        self.__agent_selector.reset()
        for _ in range(self.__config["tasks"]):
            agent = self.__agent_selector.next()
            self.__tasks[agent].append(task := self.__tasks_cards.pop())
            self.__tasks_owner[task] = agent

    def __setLogger(self, name="TheCrew", level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if not os.path.exists("logs"):
            os.mkdir("logs")
        file_handler = logging.FileHandler(f"logs/{name}-{int(time.time())}.log", mode="w")
        file_handler.setLevel(level)
        self.logger.addHandler(file_handler)
