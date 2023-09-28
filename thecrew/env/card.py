class Card:
    def __init__(self, suit, value):
        self.__suit = suit
        self.__value = value

    def suit(self):
        return self.__suit

    def value(self):
        return self.__value

    def __repr__(self):
        return f"{self.__suit}{self.__value}"


class PlayingCard(Card):
    def __init__(self, suit, value):
        super().__init__(suit, value)


class TaskCards(Card):
    def __init__(self, suit, value, player):
        super().__init__(suit, value)
        self.__player = player
