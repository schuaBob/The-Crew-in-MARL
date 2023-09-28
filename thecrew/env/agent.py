class Player:
    def __init__(self, name):
        self.__name = name
        self.__hand = []
        self.__isCommander = False

    def __repr__(self):
        return f"{self.__name}"
