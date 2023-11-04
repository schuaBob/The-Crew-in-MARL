from thecrew import thecrew_env_v0
from pettingzoo.test import api_test
seed = 65535
env = thecrew_env_v0.env(seed=seed)
env.reset()
api_test(env, num_cycles=1)
# for agent in env.agent_iter(1):
#     prev_observe, reward, terminated, truncated, info = env.last()
#     print(prev_observe)
#     print(reward)
#     print(terminated)
#     print(truncated)
#     print(info)
#     pass

env.close()
# Interact with the environment

