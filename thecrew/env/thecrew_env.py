from gymnasium.spaces import Space, Sequence, Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import random
from collections import Counter, deque
from typing import Deque
from bidict import bidict
import numpy as np

# from gymnasium.spaces import Discrete, Dict, Sequence

# TODO: need to come up a better reward map, agents can be stupid and still get reward if win for the current reward_map
REWARD_MAP = {
    "win": 50,
    "lose": -10,
    "task_complete": 10,
}


def env(**kwargs):
    return raw_env(**kwargs)


class raw_env(AECEnv):
    def __init__(
        self,
        seed: int,
        colors: int = 4,
        ranks: int = 9,
        rockets: int = 4,
        players: int = 3,
        tasks: int = 3,
    ):
        self.__config = {
            "seed": seed,
            "colors": colors,
            "ranks": ranks,
            "rockets": rockets,
            "players": players,
            "tasks": tasks,
        }

        self.possible_agents: list[str] = [f"player_{i}" for i in range(players)]
        self.__playing_cards, self.__tasks_cards = self.__generateAllCards()
        self.action_spaces: dict[str, Space] = {
            agent: Discrete(colors * ranks + rockets) for agent in self.possible_agents
        }
        # Discrete +1 becasue -1 indicates no cards played yet
        self.observation_spaces: dict[str, Space] = {
            agent: Sequence(Discrete(colors * ranks + rockets + 1, start=-1))
            for agent in self.possible_agents
        }
        print("Done init")

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        random.seed(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.__agent_selector = agent_selector(self.agents)
        self.__hands: dict[str, list[tuple[str, int]]] = {}
        self.__tasks: dict[str, list[tuple[str, int]]] = {}
        self.__suit_counters: dict[str, Counter] = {}
        self.__tasks_owner: dict[tuple[str, int], str] = {}
        self.__current_trick: list[tuple[str, tuple[str, int]]] = []
        self.__observations = {
            agent: np.full((self.__config["players"],), -1, np.int8)
            for agent in self.agents
        }

        for agent in self.agents:
            self.__hands[agent] = []
            self.__tasks[agent] = []
            self.__suit_counters[agent] = Counter()
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

    # Perfect information function. All hands returned.
    # Will need to be updated to return only the hand of the current player, as well as belief distribution over the other hands (?).
    # Also, note that we'll need to track the previously played cards as well.
    def observe(self, agent):
        return {
            "observation": self.__observations[agent],
            "action_mask": self.__legal_moves(agent),
        }

    def __legal_moves(self, agent):
        """
        Legal Moves of a agent:
        1. if trick_basecard exist, play cards with color same as trick_basecard
        2. if trick_basecard exist, play any card if player don't have cards with color same as the trick_basecard
        3. if trick_basecard not exist(first player in turn), play any cards
        """

        mask = np.zeros(len(self.playing_cards_bidict), dtype=np.int8)
        # condition 3
        if self.__agent_selector.is_first() and agent == self.agent_selection:
            for card in self.__hands[self.agent_selection]:
                mask[self.playing_cards_bidict[card]] = 1
            return mask
        # condition 2
        if (
            self.__suit_counters[agent][
                (trick_basecard := self.__current_trick[0][1])[0]
            ]
            == 0
        ):
            for card in self.__hands[self.agent_selection]:
                mask[self.playing_cards_bidict[card]] = 1
            return mask
        # condition 1
        elif self.__suit_counters[agent][trick_basecard[0]] > 0:
            mask = []
            for card in self.__hands[agent]:
                if card[0] == trick_basecard[0]:
                    mask[self.playing_cards_bidict[card]] = 1
            return mask

    # for now, action is just the index of the card to play. Hints will be added later.
    # Return value is false if the game is over, true otherwise.
    def step(self, action: int):
        """
        action: index of card to play, provided after action_masking and ideally by custom Policy
        """
        agent = self.agent_selection
        card = self.playing_cards_bidict.inverse[action]
        print(f"Current Action: {action}({card})")
        # truncation is used when num_step is set
        if self.terminations[agent] or self.truncations[agent]:
            print(f"step terminated at {agent}")
            return
        self.__hands[agent].remove(card)
        self.__suit_counters[agent][card[0]] -= 1
        self.__current_trick.append((agent, card))
        for i, trick_info in enumerate(self.__current_trick):
            self.__observations[agent][i] = self.playing_cards_bidict[trick_info[1]]
        self._cumulative_rewards[agent] = 0

        # check if trick is over
        if self.__agent_selector.is_last() and len(self.__current_trick) == len(
            self.agents
        ):
            trick_suit, trick_value = self.__current_trick[0][1]
            trick_owner = self.__current_trick[0][0]
            for card_player, (card_suit, card_value) in self.__current_trick[1:]:
                if card_suit == trick_suit and card_value > trick_value:
                    trick_value = card_value
                    trick_owner = card_player
                elif card_suit == "R" and trick_suit != "R":
                    trick_suit = "R"
                    trick_value = card[1]
                    trick_owner = card_player

            # check if any task is completed
            for _, card in self.__current_trick:
                if card in self.__tasks_owner.keys():
                    task_owner = self.__tasks_owner[card]
                    if task_owner == trick_owner:
                        self.__tasks[task_owner].remove(card)
                        self.__tasks[task_owner].sort()
                        self.__tasks_owner.pop(card)
                        # currently only reward trick_owner(task_owner)
                        self.rewards[trick_owner] += REWARD_MAP["task_complete"]
                        # Terminate if all tasks are completed
                        if len(self.__tasks_owner.keys()) == 0:
                            for agent in self.agents:
                                self.rewards[agent] += REWARD_MAP["win"]
                                self.terminations[agent] = True
                            break
                    else:
                        # currently only punish trick_owner(task_owner)
                        self.rewards[trick_owner] += REWARD_MAP["lose"]
                        # Terminate if task_owner != trick_owner
                        for agent in self.agents:
                            self.terminations[agent] = True
                        break

            self.__reinit_agents_order(trick_owner)
            self.__current_trick = []

        self.agent_selection = self.__agent_selector.next()
        self._accumulate_rewards()

    def render(self):
        s = f"config: {self.__config} \n"
        s += f"hands: {self.__hands} \n"
        s += f"tasks: {self.__tasks_owner} \n"
        s += f"commander: {self.commander}\n"
        s += f"current_trick: {self.__current_trick}\n"
        s += f"current_turn: {self.player_to_play}\n"
        return s

    def config(self):
        print(self.__config)
        print(self.possible_agents)
        print(self.__playing_cards)
        print(self.__tasks_cards)

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
        self.playing_cards_bidict = bidict(
            {k: idx for idx, k in enumerate(playing_cards)}
        )
        self.task_cards_bidict = bidict({k: idx for idx, k in enumerate(task_cards)})
        return playing_cards, task_cards

    def __deal_playing_cards(self):
        random.shuffle(self.__playing_cards)
        for card in self.__playing_cards:
            agent = self.__agent_selector.next()
            self.__hands[agent].append(card)
            self.__suit_counters[agent][card[0]] += 1

    def __reinit_agents_order(self, start_agent):
        """Reset order given a start agent"""
        idx = self.agents.index(start_agent)
        new_order = self.agents[idx:] + self.agents[0:idx]
        self.__agent_selector.reinit(new_order)

    # Random for now
    def __deal_task_cards(self):
        random.shuffle(self.__tasks_cards)
        for _ in range(self.__config["tasks"]):
            agent = self.__agent_selector.next()
            self.__tasks[agent].append(task := self.__tasks_cards.pop())
            self.__tasks_owner[task] = agent

    def action_space(self, agent: str) -> Space:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> Space:
        return self.observation_spaces[agent]
