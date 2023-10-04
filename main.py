from thecrew import thecrew_env_v0
from pettingzoo.test import api_test

env = thecrew_env_v0.env()
env.reset()
api_test(env, num_cycles=1)
for agent in env.agent_iter(10):
    
    pass

env.close()
# Interact with the environment

