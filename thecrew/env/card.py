from abc import ABC, abstractmethod


class Card(ABC):
    @abstractmethod
    def __repr__(self):
        pass


class PlayingCard(Card):
    def __init__(self, suit, value):
        self.__suit = suit
        self.__value = value

    def suit(self):
        return self.__suit

    def value(self):
        return self.__value

    def __repr__(self):
        return f"{self.__suit}{self.__value}"


class TaskCards(Card):
    def __init__(self, suit, value):
        self.__suit = suit
        self.__value = value
        self.__player = None

    def suit(self):
        return self.__suit

    def value(self):
        return self.__value

    def __repr__(self):
        return f"{self.__suit}{self.__value}"


class ReminderCards(Card):
    pass
