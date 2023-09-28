from thecrew import thecrew_env_v0
from thecrew.env.agent import Player

env = thecrew_env_v0.env()
# env.config()
agents = [Player(name) for name in env.possible_agents]
# Interact with the environment
