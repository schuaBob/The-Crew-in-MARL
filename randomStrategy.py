from thecrew import thecrew_env_v0
from argparse import ArgumentParser


def main(**kwargs):
    env = thecrew_env_v0.env(render=kwargs["render"])
    env.reset(seed=kwargs["seed"])
    for agent in env.agent_iter():
        prev_observe, reward, terminated, truncated, info = env.last()
        action = None
        if terminated or truncated:
            action = None
        else:
            action = env.action_space(agent).sample(mask=prev_observe["action_mask"])
        env.step(action)

    env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=65535)
    parser.add_argument("--render", action="store_false")
    args = parser.parse_args()
    main(seed=args.seed, render=args.render)
