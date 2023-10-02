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

    def __cmp__(self, other):
        if self.suit() == other.suit():
            return self.value() - other.value()
        return ord(self.suit()) - ord(other.suit())
    

    def __lt__(self, other):
        return self.__cmp__(other) < 0
    
    def __le__(self, other):
        return self.__cmp__(other) <= 0
    
    def __eq__(self, other):
        return self.__cmp__(other) == 0
    
    def __ne__(self, other):
        return self.__cmp__(other) != 0
    
    def __gt__(self, other):
        return self.__cmp__(other) > 0
    
    def __ge__(self, other):
        return self.__cmp__(other) >= 0
    
    def __hash__(self):
        return hash((self.__suit, self.__value))

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

if __name__ == "__main__":
    a = PlayingCard("R", 1)
    b = PlayingCard("R", 2)
    c = PlayingCard("G", 3)
    print(a < b)
    print(c > b)