import random
import numpy as np
import gym
from gym import spaces
from copy import copy

class CustomCrewGymEnvironment(gym.Env):
    def __init__(self):
        super(CustomCrewGymEnvironment, self).__init__()
        # Define game-specific constants
        self.num_colors = 4
        self.num_ranks = 9
        self.num_rockets = 4
        self.num_players = 3
        self.max_actions = self.num_colors * self.num_ranks + self.num_rockets
        self.total_target_missions = 1  # Total number of target missions
        self.rounds = 0 # Total number of rounds played
        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]

        # Calculate the size of the observation space
        self.player_card_encoding_size = (self.num_colors * self.num_ranks + self.num_rockets) * self.num_players
        self.task_card_encoding_size = (self.num_colors * self.num_ranks)*self.num_players
        self.commander_encoding_size = self.num_players
        self.starting_color_encoding_size = self.num_colors+1 #+1 for no starting color

        self.action_space = spaces.Discrete(self.max_actions)
        
        self.observation_space = spaces.Dict({
            'observation': spaces.MultiBinary(self.player_card_encoding_size + self.task_card_encoding_size + self.commander_encoding_size + self.starting_color_encoding_size),
            'action_mask': spaces.MultiBinary(self.max_actions)
        })
        
        #self.observation_space =  spaces.MultiBinary(self.player_card_encoding_size + self.task_card_encoding_size + self.commander_encoding_size + self.starting_color_encoding_size)
        self.action_mask = spaces.MultiBinary(self.max_actions)

        # Initialize game state
        self.deck = [(color, rank) for color in range(self.num_colors) for rank in range(self.num_ranks)] + [(4, rank) for rank in range(self.num_rockets)]
        self.hands = self.distribute_cards()
        self.target_missions = [self.assign_target() for _ in range(self.total_target_missions)]  # Assign target cards to players
         # Randomly assign the commander

    def reset(self):
        # Reset the game state
        self.deck = [(color, rank) for color in range(self.num_colors) for rank in range(self.num_ranks)] + [(4, rank) for rank in range(self.num_rockets)]
        self.hands = self.distribute_cards()
        self.target_missions = [self.assign_target() for _ in range(self.total_target_missions)]  # Assign target cards to players
        self.commander = random.randint(0, self.num_players - 1)
        self.current_player = self.commander
        self.starting_color = -1  # -1 is no starting color

        # Initialize the highest card played to None (no card played yet)
        self.highest_rocket_played = -1
        self.highest_card_played = -1
        self.current_trick = {}
        self.winning_card = None
        self.rounds = 0

        return self.observe(self.current_player)

    def distribute_cards(self):
        # Make a copy of the deck
        deck_copy = self.deck.copy()

        # Shuffle the entire copy of the deck
        random.shuffle(deck_copy)

        # Calculate the base number of cards for each player
        total_cards = len(deck_copy)
        num_players = self.num_players
        base_cards = total_cards // num_players

        # Calculate the number of remaining cards
        remaining_cards = total_cards % num_players

        # Initialize hands with self.num_players number of sublists
        hands = [[] for _ in range(num_players)]

        # Distribute base_cards evenly
        for i in range(base_cards):
            for player_idx in range(num_players):
                hands[player_idx].append(deck_copy.pop())

        # Distribute remaining cards randomly
        for i in range(remaining_cards):
            player_idx = random.randint(0, num_players - 1)
            hands[player_idx].append(deck_copy.pop())

        return hands

    def assign_target(self):
        # Randomly assign the target card to a player from self.deck, excluding rockets (color 4)
        target_player = random.randint(0, self.num_players - 1)
        target_card = random.choice([card for card in self.deck if card[0] != self.num_colors])
        return target_card, target_player

    def observe(self, agent):
        # Construct the observation space as described
        observation = np.zeros(self.observation_space['observation'].shape, dtype=int)

        # Fill in player's cards as one-hot encoding
        for player_idx, player_hand in enumerate(self.hands):
            for card in player_hand:
                color, rank = card
                player_offset = player_idx * (self.num_colors * self.num_ranks + self.num_rockets)
                card_offset = color * self.num_ranks + rank
                index = card_offset + player_offset
                observation[index] = 1

        # Fill in remaining task missions
        # self.target_missions has number of remaining missions and has the form [(color, rank), player_idx]
        
        for ele in self.target_missions:
            color, rank = ele[0]
            player_idx = ele[1]
            player_offset = player_idx * (self.num_colors * self.num_ranks) #mission cards cant be rockets
            card_offset = color * self.num_ranks + rank
            index = card_offset + player_offset + self.num_players * (self.num_colors * self.num_ranks+ self.num_rockets)
            observation[index] = 1

        # Fill in commander and starting color
        observation[self.player_card_encoding_size + self.task_card_encoding_size+ self.commander] = 1  # One-hot encoding for commander
        
        if self.starting_color==-1:
            observation[-1]=1
        else:
            observation[self.player_card_encoding_size + self.task_card_encoding_size+ self.commander_encoding_size + self.starting_color] = 1  # One-hot encoding for starting color

        # Construct action mask
        action_mask = self.get_legal_moves(agent,observation) 

        return {'observation': observation, 'action_mask': action_mask}

    def get_legal_moves(self, agent, observation):
        # Define logic to get legal moves based on game state
        action_mask = np.zeros(self.action_mask.shape, dtype=int)
        agent_hand = self.hands[agent]
        possible_options = []
        if np.max(observation[-5:-1]) == 1:
            # Find the cards that the agent player has with the starting color
            starting_color = np.argmax(observation[-5:-1])
            for hands in agent_hand:
                colour,rank = hands
                if colour == starting_color:
                    possible_options.append(hands)
                    action_mask[colour*self.num_ranks+rank] = 1
        if possible_options == []:
            # If the starting color is not specified (-1), all actions are legal
            for hands in agent_hand:
                colour,rank = hands
                action_mask[colour*self.num_ranks+rank] = 1
        return action_mask

    def step(self, action):
        # Execute the player's action
        reward = 0
        done = None

        # Handle other actions
        color, rank = divmod(action, self.num_ranks)  # Decode the action to color and rank
        played_card = (color, rank)

        print(f"Player {self.current_player} played {played_card}")
        # Remove the played card from the player's hand
        if played_card in self.hands[self.current_player]:
            self.hands[self.current_player].remove(played_card)
            self.current_trick[played_card]= self.current_player

        if self.starting_color == -1:
            self.highest_card_played = rank
            self.starting_color = color
            self.winning_player = self.current_player
            self.winning_card = played_card

        if color == 4:
            # Rocket is played, update the highest rocket played
            if rank > self.highest_rocket_played:
                self.highest_rocket_played = rank
                self.winning_player = self.current_player
                self.winning_card = played_card
        elif color == self.starting_color:
            # Update the highest card played if the color matches the starting color
            if rank > self.highest_card_played:
                self.highest_card_played = rank
                self.winning_player = self.current_player
                self.winning_card = played_card
                
        # Check if all players have played
        if self.current_player == (self.commander-1)%self.num_players:
            print ("-"*50)
            if self.rounds > (self.max_actions//self.num_players+1):
                print("Game Over! All rounds have been played!")
                reward -= 100
                done = False
                return self.observe(self.current_player), reward, done, {}
            
            print(f"Round Summary:{self.rounds}")
            print(f"tricks played: {self.current_trick}")
            print(f"Player {self.winning_player} won the trick!")
            print ("-"*50)
            # Determine the winner of the trick
            if self.winning_player is not None:
               
                # Check if the winning card matches a target mission
                for mission, target_player in self.target_missions:
                    print(f"Mission: {mission}, Target Player: {target_player}")
                    print(f"Current Trick: {self.current_trick}")
                    if mission in self.current_trick and target_player == self.winning_player:
                        #import pdb;pdb.set_trace();
                        reward += 50  # Player wins a target mission
                        print(f"Congratulations! Target Player won a target mission!")
                        print ("-"*50)
                        self.target_missions.remove((mission, target_player))  # Remove the mission from the list
                    elif mission in self.current_trick and target_player != self.winning_player:
                        #import pdb;pdb.set_trace();
                        reward -= 100  # Player played a target mission but did not win
                        done = False
                        print(f"Game Over! We lost a target mission!")
                        print ("-"*50)
                        return self.observe(self.current_player), reward, done, {}
                        
                # reset all variables for next turn
                self.commander = self.winning_player  # The winner becomes the new commander
                self.current_player = self.commander  # The commander starts the next round
                self.starting_color = -1  # Reset the starting color
                self.highest_card_played = -1  # Reset the highest card played
                self.highest_rocket_played = -1  # Reset the highest rocket played
                self.current_trick = {}
                self.rounds+=1
                self.winning_card = None
        else:
            # Update the current player for the next turn
            self.current_player = (self.current_player + 1) % self.num_players    
            # Check if all target missions have been completed and won by assigned players
            if not self.target_missions:
                print("All target missions have been completed!")
                done = True
                reward += 100
                
        return self.observe(self.current_player), reward, done, {}


    def render(self, mode='human'):
        #print("Round: ",self.rounds)
        # print(f"self.hands[0]- cards ={len(self.hands[0])}: ",self.hands[0])
        # print(f"self.hands[1]- cards ={len(self.hands[1])}: ",self.hands[1])
        # print(f"self.hands[2]- cards ={len(self.hands[2])}: ",self.hands[2])
        # print(f"Commander: Player {self.commander}")
        # print(f"Starting Color: {self.starting_color}")
        # print(f"Highest Card Played: {self.highest_card_played}")
        # print(f"Highest Rocket Played: {self.highest_rocket_played}")
        # print(f"current trick: {self.current_trick}")
        # print(f"Remaining Missions: {self.target_missions}")
        # print(f"current reward: {reward}")
        print('-' * 50)
        
    def close(self):
        pass

if __name__ == "__main__":
    win = 0
    loss = 0 
    for i in range(100):
        env = CustomCrewGymEnvironment()
        obs = env.reset()
        done = None
        for player in range(env.num_players):
            print(f"Player's Hand (Player {player}) - {(len(env.hands[player]))}: {env.hands[player]}")
        print('-' * 50)
        reward_sum = 0    
        for _ in range(100):
            print(f"Current Player: {env.current_player}")
            current_player = env.current_player
            action_mask = obs['action_mask']

            # Sample a legal action based on the action_mask
            legal_actions = [i for i in range(len(action_mask)) if action_mask[i] == 1]
            if legal_actions:
                action = random.choice(legal_actions)
            else:
                # If there are no legal actions, choose a random action
                print("No legal actions, recheck implementation")
                #action = random.randint(0, env.max_actions - 1)
                done = False
                loss +=1
                
            if done == None:
                obs, reward, done, _ = env.step(action)
                env.render()        
                reward_sum += reward
                # print(f"Reward: {reward}")
                # print(f"Done: {done}")
                # print('-' * 50)
                if done==True:
                    win +=1
                elif done==False:
                    loss +=1
                
            if done!=None:
                print ("-"*50)
                print(f"Reward: {reward_sum}")
                print(f"Done: {done}")
                print ("-"*50)
                break
    
    print(f"Total Wins: {win}")
    print(f"Total Loss: {loss}")
    print(f"win-rate: {win/(win+loss)*100}%")

